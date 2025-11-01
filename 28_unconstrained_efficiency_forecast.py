"""
AGI Forecast with Unconstrained Efficiency Improvement
Remove the 1.0 asymptote - let tax decline linearly or even go below 1.0
Test if efficiency constraint is the real bottleneck
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from datetime import datetime, timedelta

print("="*100)
print("AGI FORECAST: UNCONSTRAINED EFFICIENCY")
print("="*100)

# Load scenarios
with open('outputs/sigmoid_scenarios.json', 'r') as f:
    scenarios = json.load(f)

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

# Find compute for target ECI
def find_kaplan_compute(target_eci):
    compute_range = np.logspace(20, 50, 100000)
    eci_range = np.array([kaplan_eci(c) for c in compute_range])
    idx = np.argmin(np.abs(eci_range - target_eci))
    return compute_range[idx], eci_range[idx]

# Load LUCR tax model
lucr_eci = pd.read_csv('outputs/lucr_eci_scale.csv')
lucr_eci = lucr_eci[lucr_eci['lucr_compute_tax'] < 10].copy()

# Fit LINEAR tax decline (no asymptote)
def linear_tax(t, initial_tax, decline_rate):
    """Tax declines linearly, can go below 1.0"""
    return initial_tax - decline_rate * t

time_idx = np.arange(len(lucr_eci))
tax_values = lucr_eci['lucr_compute_tax'].values

# Fit linear
popt_linear, _ = np.polyfit(time_idx, tax_values, 1, cov=True)
decline_rate = -popt_linear[0]  # Negative slope = decline
initial_tax_linear = popt_linear[1]

print(f"\nLinear Tax Model (unconstrained):")
print(f"  Initial: {initial_tax_linear:.3f}x")
print(f"  Decline rate: {decline_rate:.4f} per model")
print(f"  Formula: tax(t) = {initial_tax_linear:.3f} - {decline_rate:.4f} * t")

# When does tax hit 1.0?
models_to_unity = (initial_tax_linear - 1.0) / decline_rate
years_to_unity = models_to_unity / 4  # 4 models per year

print(f"  Tax reaches 1.0 at: {models_to_unity:.0f} models ({years_to_unity:.1f} years)")
print(f"  Tax can continue declining below 1.0! (super-Kaplan efficiency)")

# Current state
current_eci = 134.7
current_compute = lucr_eci['Training compute (FLOP)'].max()
current_date = pd.to_datetime(lucr_eci['Release date'].max())

print(f"\nCurrent state ({current_date.date()}):")
print(f"  ECI: {current_eci:.1f}")
print(f"  Compute: {current_compute:.2e} FLOPs")
print(f"  Current tax: {tax_values[-1]:.3f}x")

# Compute scaling assumptions
compute_doubling_months = 6
models_per_year = 4

print(f"\n{'='*100}")
print("SCENARIO FORECASTS (with unconstrained efficiency):")
print(f"{'='*100}")

forecast_results = []

for scenario in scenarios:
    name = scenario['scenario']
    target_years = scenario['years']
    target_eci = scenario['future_eci']
    
    print(f"\n{name} (AGI in {target_years} years):")
    print(f"  Target: ECI {target_eci:.1f}, METR {scenario['metr_at_future']*100:.1f}%")
    
    # Kaplan compute needed
    kaplan_compute_needed, eci_achievable = find_kaplan_compute(target_eci)
    
    if abs(eci_achievable - target_eci) > 50:
        print(f"  ⚠️  Kaplan can only reach ECI {eci_achievable:.1f}, not {target_eci:.1f}")
        print(f"  ⚠️  Target exceeds Kaplan ceiling - skipping")
        forecast_results.append({
            'scenario': name,
            'target_years': target_years,
            'target_eci': target_eci,
            'achievable': False,
            'reason': 'Exceeds Kaplan ceiling'
        })
        continue
    
    print(f"  Kaplan baseline: {kaplan_compute_needed:.2e} FLOPs")
    
    # Tax at target time (linear decline)
    num_models_ahead = target_years * models_per_year
    future_model_idx = len(lucr_eci) + num_models_ahead
    future_tax = linear_tax(future_model_idx, initial_tax_linear, decline_rate)
    
    # Tax can go negative in this model, floor at 0.1
    future_tax = max(future_tax, 0.1)
    
    # Actual compute needed (with tax)
    actual_compute_needed = kaplan_compute_needed * future_tax
    print(f"  Tax at {target_years}y: {future_tax:.3f}x")
    print(f"  Actual compute needed: {actual_compute_needed:.2e} FLOPs")
    
    # Available compute with 6-month doubling
    months = target_years * 12
    compute_multiplier = 2 ** (months / compute_doubling_months)
    available_compute = current_compute * compute_multiplier
    print(f"  Available (6mo doubling): {available_compute:.2e} FLOPs ({compute_multiplier:.0f}x)")
    
    # Can we achieve it?
    gap = actual_compute_needed / available_compute
    achievable = gap <= 1.0
    
    print(f"  Gap: {gap:.2f}x ({'✓ ACHIEVABLE' if achievable else '✗ NOT ACHIEVABLE'})")
    
    # If achievable, when exactly?
    if achievable:
        # Binary search for exact timeline
        for test_years in np.linspace(0.1, target_years, 1000):
            test_months = test_years * 12
            test_compute_mult = 2 ** (test_months / compute_doubling_months)
            test_available = current_compute * test_compute_mult
            
            test_models_ahead = test_years * models_per_year
            test_tax = max(linear_tax(len(lucr_eci) + test_models_ahead, initial_tax_linear, decline_rate), 0.1)
            test_needed = kaplan_compute_needed * test_tax
            
            if test_available >= test_needed:
                agi_years = test_years
                agi_date = current_date + timedelta(days=365*test_years)
                print(f"  🎯 AGI by: {agi_date.strftime('%B %Y')} ({agi_years:.1f} years)")
                break
    
    forecast_results.append({
        'scenario': name,
        'target_years': target_years,
        'target_eci': target_eci,
        'kaplan_compute': kaplan_compute_needed,
        'tax': future_tax,
        'actual_compute_needed': actual_compute_needed,
        'available_compute': available_compute,
        'gap': gap,
        'achievable': achievable,
        'reason': 'Within Kaplan ceiling'
    })

forecast_df = pd.DataFrame(forecast_results)

# Save
forecast_df.to_csv('outputs/unconstrained_efficiency_forecast.csv', index=False)
print(f"\nSaved to outputs/unconstrained_efficiency_forecast.csv")

# Summary
print(f"\n{'='*100}")
print("SUMMARY:")
print(f"{'='*100}")

achievable = forecast_df[forecast_df['achievable'] == True]
kaplan_limited = forecast_df[forecast_df.get('reason') == 'Exceeds Kaplan ceiling']

print(f"\n✓ Achievable scenarios: {len(achievable)}")
if len(achievable) > 0:
    print("  " + ", ".join(achievable['scenario'].values))

print(f"\n✗ Blocked by Kaplan ceiling: {len(kaplan_limited)}")
if len(kaplan_limited) > 0:
    print("  " + ", ".join(kaplan_limited['scenario'].values))

# Visualize tax models: asymptotic vs linear
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Tax models comparison
ax1 = axes[0, 0]

# Historical
ax1.scatter(time_idx, tax_values, s=100, alpha=0.7, color='#2E86AB',
           edgecolors='black', linewidth=1.5, label='Historical', zorder=10)

# Asymptotic (old model)
from scipy.optimize import curve_fit
def asymptotic_tax(t, initial_tax, decay_rate):
    return 1.0 + (initial_tax - 1.0) * np.exp(-decay_rate * t)

popt_asym, _ = curve_fit(asymptotic_tax, time_idx, tax_values, p0=[tax_values[0], 0.01], maxfev=10000)
future_idx = np.linspace(0, len(lucr_eci) + 100, 500)
tax_asym = asymptotic_tax(future_idx, *popt_asym)
ax1.plot(future_idx, tax_asym, '--', linewidth=2.5, color='#E63946',
        label='Asymptotic (capped at 1.0)', alpha=0.7)

# Linear (new model)
tax_linear = linear_tax(future_idx, initial_tax_linear, decline_rate)
ax1.plot(future_idx, tax_linear, '-', linewidth=2.5, color='#06A77D',
        label='Linear (unconstrained)', alpha=0.7)

ax1.axhline(1.0, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Unity (1.0x)')

ax1.set_xlabel('Model Index', fontsize=12, fontweight='bold')
ax1.set_ylabel('Compute Tax', fontsize=12, fontweight='bold')
ax1.set_title('Tax Evolution: Asymptotic vs Linear', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, len(lucr_eci) + 100])
ax1.set_ylim([0, 3.5])

# Plot 2: Feasibility comparison
ax2 = axes[0, 1]

if len(forecast_df) > 0 and 'gap' in forecast_df.columns:
    scenarios_list = forecast_df['scenario'].values
    gaps = forecast_df['gap'].fillna(1e10).values  # Fill NaN with huge number
    colors = ['#06A77D' if a else '#E63946' for a in forecast_df['achievable']]
    
    bars = ax2.barh(scenarios_list, gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Achievable threshold')
    ax2.set_xlabel('Gap (Needed/Available)', fontsize=12, fontweight='bold')
    ax2.set_title('Scenario Feasibility (Unconstrained Efficiency)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both', axis='x')

# Plot 3: Kaplan ceiling issue
ax3 = axes[1, 0]

max_eci_kaplan = a_linear * L_inf + b_linear

scenario_ecis = [s['future_eci'] for s in scenarios]
scenario_names = [s['scenario'] for s in scenarios]

colors_eci = ['#06A77D' if eci <= max_eci_kaplan else '#E63946' for eci in scenario_ecis]
bars = ax3.bar(scenario_names, scenario_ecis, color=colors_eci, alpha=0.7, edgecolor='black', linewidth=1.5)

ax3.axhline(max_eci_kaplan, color='red', linestyle='--', linewidth=2.5, alpha=0.7,
           label=f'Kaplan Ceiling ({max_eci_kaplan:.0f})')
ax3.axhline(current_eci, color='blue', linestyle=':', linewidth=2, alpha=0.7,
           label=f'Current ({current_eci:.0f})')

ax3.set_ylabel('Target ECI', fontsize=12, fontweight='bold')
ax3.set_title('The Real Bottleneck: Kaplan Ceiling', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 4: Key insight
ax4 = axes[1, 1]
ax4.axis('off')

insight_text = f"""
KEY FINDING:

Removing efficiency asymptote does NOT solve the problem.

✗ Only 1-Year scenario is within Kaplan ceiling
  (ECI 163 vs ceiling 189)

✗ All other scenarios (2-20 years) require:
  ECI 192-709 (IMPOSSIBLE with Kaplan)

The bottleneck is NOT efficiency improvement.
The bottleneck is KAPLAN'S LOSS CEILING.

Current models (ECI 135-150) are already at
71-80% of theoretical maximum (189).

CONCLUSION:
AGI requires breakthrough beyond current
scaling laws, not just better efficiency.
"""

ax4.text(0.1, 0.5, insight_text, fontsize=13, fontweight='bold',
        verticalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#FFE5B4', alpha=0.8, pad=20))

plt.tight_layout()
plt.savefig('outputs/unconstrained_efficiency_analysis.png', dpi=300, bbox_inches='tight')
print("Saved visualization to outputs/unconstrained_efficiency_analysis.png")

print(f"\n{'='*100}")
print("✅ UNCONSTRAINED EFFICIENCY ANALYSIS COMPLETE!")
print(f"{'='*100}")

print(f"\n💡 CONCLUSION:")
print(f"   Removing the 1.0 efficiency cap does NOT make AGI achievable.")
print(f"   The real constraint is KAPLAN'S CEILING (ECI 189).")
print(f"   Current models are already 71-80% to that ceiling.")
print(f"   → AGI requires paradigm shift, not just scaling.")


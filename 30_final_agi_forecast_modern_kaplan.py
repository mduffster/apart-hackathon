"""
Final AGI Forecast with Modern Scaling Law
Using ECI = 278.9 - 1000 * C^(-0.0331) from 2023-2025 data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from datetime import datetime, timedelta

print("="*100)
print("FINAL AGI FORECAST: MODERN SCALING LAW")
print("="*100)

# Load modern scaling law
with open('outputs/modern_scaling_law.json', 'r') as f:
    modern_law = json.load(f)

print(f"\nModern Scaling Law ({modern_law['method']}):")
print(f"  {modern_law['formula']}")
print(f"  R² = {modern_law['r2']:.4f}")
print(f"  Data: {modern_law['n_models']} models from {modern_law['data_source']}")

# Extract parameters
eci_inf, A_eci, alpha_eci = modern_law['params']

def modern_kaplan_eci(compute):
    """ECI = eci_inf - A * C^(-alpha)"""
    return eci_inf - A_eci * np.power(compute, -alpha_eci)

# Find compute for target ECI
def find_modern_compute(target_eci):
    """Solve: target_eci = eci_inf - A * C^(-alpha) for C"""
    if target_eci >= eci_inf:
        return np.inf, eci_inf
    
    # C = (A / (eci_inf - target_eci))^(1/alpha)
    compute = np.power(A_eci / (eci_inf - target_eci), 1.0 / alpha_eci)
    return compute, target_eci

print(f"\nCeiling: ECI {eci_inf:.1f}")

# Load scenarios
with open('outputs/sigmoid_scenarios.json', 'r') as f:
    scenarios = json.load(f)

# Load LUCR tax (asymptotic)
lucr_eci = pd.read_csv('outputs/lucr_eci_scale.csv')
lucr_eci = lucr_eci[lucr_eci['lucr_compute_tax'] < 10].copy()

def asymptotic_tax(t, initial_tax, decay_rate):
    return 1.0 + (initial_tax - 1.0) * np.exp(-decay_rate * t)

time_idx = np.arange(len(lucr_eci))
tax_values = lucr_eci['lucr_compute_tax'].values
popt, _ = curve_fit(asymptotic_tax, time_idx, tax_values, p0=[tax_values[0], 0.01], maxfev=10000)
initial_tax_fit, decay_rate_fit = popt

print(f"\nLUCR Tax Model (asymptotic):")
print(f"  Initial: {initial_tax_fit:.3f}x")
print(f"  Decay: {decay_rate_fit:.4f} per model")
print(f"  Approaches: 1.0x")

# Current state
current_eci = 134.7
current_compute = lucr_eci['Training compute (FLOP)'].max()
current_date = pd.to_datetime(lucr_eci['Release date'].max())

print(f"\nCurrent state ({current_date.date()}):")
print(f"  ECI: {current_eci:.1f} ({(current_eci/eci_inf)*100:.1f}% to ceiling)")
print(f"  Compute: {current_compute:.2e} FLOPs")

# Compute scaling
compute_doubling_months = 6
models_per_year = 4

print(f"\n{'='*100}")
print("SCENARIO FORECASTS:")
print(f"{'='*100}")

forecast_results = []

for scenario in scenarios:
    name = scenario['scenario']
    target_years = scenario['years']
    target_eci = scenario['future_eci']
    
    print(f"\n{name} (AGI in {target_years} years):")
    print(f"  Target: ECI {target_eci:.1f}, METR {scenario['metr_at_future']*100:.1f}%")
    
    # Check if achievable with new scaling law
    if target_eci >= eci_inf:
        print(f"  ✗ EXCEEDS CEILING (ECI {eci_inf:.1f})")
        forecast_results.append({
            'scenario': name,
            'target_years': target_years,
            'target_eci': target_eci,
            'achievable': False,
            'reason': 'Exceeds ceiling'
        })
        continue
    
    # Compute needed
    compute_needed, eci_achievable = find_modern_compute(target_eci)
    print(f"  Modern scaling: {compute_needed:.2e} FLOPs")
    
    # Tax at target time
    num_models_ahead = target_years * models_per_year
    future_model_idx = len(lucr_eci) + num_models_ahead
    future_tax = asymptotic_tax(future_model_idx, initial_tax_fit, decay_rate_fit)
    
    # Actual compute with tax
    actual_compute_needed = compute_needed * future_tax
    print(f"  Tax at {target_years}y: {future_tax:.3f}x")
    print(f"  Actual needed: {actual_compute_needed:.2e} FLOPs")
    
    # Available compute
    months = target_years * 12
    compute_multiplier = 2 ** (months / compute_doubling_months)
    available_compute = current_compute * compute_multiplier
    print(f"  Available (6mo doubling): {available_compute:.2e} FLOPs ({compute_multiplier:.0f}x)")
    
    # Gap
    gap = actual_compute_needed / available_compute
    achievable = gap <= 1.0
    
    print(f"  Gap: {gap:.2f}x ({'✓ ACHIEVABLE' if achievable else '✗ NOT ACHIEVABLE'})")
    
    # If achievable, find exact date
    if achievable:
        for test_years in np.linspace(0.1, target_years, 1000):
            test_months = test_years * 12
            test_compute_mult = 2 ** (test_months / compute_doubling_months)
            test_available = current_compute * test_compute_mult
            
            test_models_ahead = test_years * models_per_year
            test_tax = asymptotic_tax(len(lucr_eci) + test_models_ahead, initial_tax_fit, decay_rate_fit)
            test_needed = compute_needed * test_tax
            
            if test_available >= test_needed:
                agi_years = test_years
                agi_date = current_date + timedelta(days=365*test_years)
                print(f"  🎯 AGI by: {agi_date.strftime('%B %Y')} ({agi_years:.1f} years)")
                
                forecast_results.append({
                    'scenario': name,
                    'target_years': target_years,
                    'target_eci': target_eci,
                    'compute_needed': compute_needed,
                    'tax': future_tax,
                    'actual_needed': actual_compute_needed,
                    'available': available_compute,
                    'gap': gap,
                    'achievable': True,
                    'agi_years': agi_years,
                    'agi_date': agi_date.strftime('%Y-%m')
                })
                break
    else:
        forecast_results.append({
            'scenario': name,
            'target_years': target_years,
            'target_eci': target_eci,
            'compute_needed': compute_needed,
            'tax': future_tax,
            'actual_needed': actual_compute_needed,
            'available': available_compute,
            'gap': gap,
            'achievable': False,
            'reason': 'Insufficient compute'
        })

forecast_df = pd.DataFrame(forecast_results)
forecast_df.to_csv('outputs/final_agi_forecast_modern.csv', index=False)

print(f"\n{'='*100}")
print("SUMMARY:")
print(f"{'='*100}")

achievable = forecast_df[forecast_df['achievable'] == True]
print(f"\n✓ ACHIEVABLE SCENARIOS: {len(achievable)}")
if len(achievable) > 0:
    for _, row in achievable.iterrows():
        print(f"  • {row['scenario']:10s}: AGI by {row.get('agi_date', 'N/A')} ({row.get('agi_years', 0):.1f} years)")

ceiling_blocked = forecast_df[forecast_df.get('reason') == 'Exceeds ceiling']
print(f"\n✗ BLOCKED BY CEILING: {len(ceiling_blocked)}")
if len(ceiling_blocked) > 0:
    print(f"  " + ", ".join(ceiling_blocked['scenario'].values))

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Scaling law comparison (old vs new)
ax1 = axes[0, 0]

compute_range = np.logspace(24, 32, 500)

# Old Kaplan (via linear loss→ECI)
old_L_inf = 1.69
old_A = 1.69
old_alpha = 0.076
old_a_linear = -2748.75
old_b_linear = 4834.07

old_loss = old_L_inf + old_A * np.power(compute_range, -old_alpha)
old_eci = old_a_linear * old_loss + old_b_linear

# New modern fit
new_eci = modern_kaplan_eci(compute_range)

ax1.plot(compute_range, old_eci, '--', linewidth=3, color='#E63946',
        label='Old Kaplan (2020)', alpha=0.7)
ax1.plot(compute_range, new_eci, '-', linewidth=3, color='#06A77D',
        label='Modern Fit (2023-2025)', alpha=0.9)

ax1.axhline(188.7, color='#E63946', linestyle=':', linewidth=2, alpha=0.5)
ax1.axhline(eci_inf, color='#06A77D', linestyle=':', linewidth=2, alpha=0.5)

ax1.scatter([current_compute], [current_eci], s=300, marker='*',
           color='gold', edgecolors='black', linewidth=2, zorder=15,
           label='Current')

ax1.set_xscale('log')
ax1.set_xlabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax1.set_ylabel('ECI Score', fontsize=12, fontweight='bold')
ax1.set_title('Scaling Law Comparison: 48% Higher Ceiling!', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_ylim([100, 300])

# Plot 2: Scenario feasibility
ax2 = axes[0, 1]

if len(forecast_df) > 0:
    scenarios_list = forecast_df['scenario'].values
    colors = ['#06A77D' if a else '#E63946' for a in forecast_df['achievable']]
    
    # Show gap for compute-limited scenarios
    gaps = []
    for _, row in forecast_df.iterrows():
        if row.get('reason') == 'Exceeds ceiling':
            gaps.append(100)  # Arbitrary high value
        else:
            gaps.append(row.get('gap', 100))
    
    bars = ax2.barh(scenarios_list, gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Achievable')
    ax2.set_xlabel('Gap (Needed/Available)', fontsize=12, fontweight='bold')
    ax2.set_title('Scenario Feasibility with Modern Scaling', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both', axis='x')

# Plot 3: Timeline to AGI
ax3 = axes[1, 0]

if len(achievable) > 0:
    agi_years = achievable['agi_years'].values
    agi_scenarios = achievable['scenario'].values
    
    bars = ax3.barh(agi_scenarios, agi_years, color='#06A77D', alpha=0.7, edgecolor='black', linewidth=2)
    
    for i, years in enumerate(agi_years):
        date_str = achievable.iloc[i]['agi_date']
        ax3.text(years + 0.1, i, f'{years:.1f}y ({date_str})', 
                va='center', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Years to AGI', fontsize=12, fontweight='bold')
    ax3.set_title('AGI Timeline for Achievable Scenarios', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='x')
else:
    ax3.text(0.5, 0.5, 'No achievable\nscenarios', ha='center', va='center',
            fontsize=16, fontweight='bold', transform=ax3.transAxes)
    ax3.axis('off')

# Plot 4: Key findings
ax4 = axes[1, 1]
ax4.axis('off')

findings_text = f"""
FINAL AGI FORECAST

Modern Scaling Law (2023-2025):
  Ceiling: ECI {eci_inf:.1f} (+48% from 2020)
  Current: ECI {current_eci:.1f} (54% to ceiling)

Achievable Scenarios: {len(achievable)}/{len(forecast_df)}
"""

if len(achievable) > 0:
    earliest = achievable.iloc[0]
    findings_text += f"""
  Earliest AGI: {earliest['agi_date']}
  ({earliest['agi_years']:.1f} years from now)
  
  Scenario: {earliest['scenario']}
  Target: ECI {earliest['target_eci']:.0f}, METR 90%
"""
else:
    findings_text += "\n  No scenarios achievable\n  with current assumptions"

findings_text += f"""
\nKey Insight:
  Modern data shows HIGHER ceiling
  than 2020 Kaplan predicted.
  
  But ceiling still exists.
  Long-term AGI (10+ years) still
  requires breakthrough beyond
  current scaling paradigm.
"""

ax4.text(0.1, 0.5, findings_text, fontsize=11, fontweight='bold',
        verticalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E6F3FF', alpha=0.9, pad=20))

plt.tight_layout()
plt.savefig('outputs/final_agi_forecast_modern.png', dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to outputs/final_agi_forecast_modern.png")

print(f"\n{'='*100}")
print("✅ FINAL AGI FORECAST COMPLETE!")
print(f"{'='*100}")


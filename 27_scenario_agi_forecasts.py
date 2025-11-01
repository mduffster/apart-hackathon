"""
AGI Forecasts for Each Sigmoid Scenario
Map ECI targets → Compute requirements (Kaplan + LUCR) → Timelines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from datetime import datetime, timedelta

print("="*100)
print("AGI FORECASTS FOR SIGMOID SCENARIOS")
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
    return compute_range[idx]

# Load LUCR tax model
lucr_eci = pd.read_csv('outputs/lucr_eci_scale.csv')
lucr_eci = lucr_eci[lucr_eci['lucr_compute_tax'] < 10].copy()

# Fit asymptotic tax
def asymptotic_tax(t, initial_tax, decay_rate):
    return 1.0 + (initial_tax - 1.0) * np.exp(-decay_rate * t)

time_idx = np.arange(len(lucr_eci))
tax_values = lucr_eci['lucr_compute_tax'].values
popt, _ = curve_fit(asymptotic_tax, time_idx, tax_values, p0=[tax_values[0], 0.01], maxfev=10000)
initial_tax_fit, decay_rate_fit = popt

print(f"\nLUCR Tax Model:")
print(f"  Initial: {initial_tax_fit:.3f}x")
print(f"  Decay rate: {decay_rate_fit:.4f} per model")

# Current state
current_eci = 134.7
current_compute = lucr_eci['Training compute (FLOP)'].max()
current_date = pd.to_datetime(lucr_eci['Release date'].max())

print(f"\nCurrent state ({current_date.date()}):")
print(f"  ECI: {current_eci:.1f}")
print(f"  Compute: {current_compute:.2e} FLOPs")

# Compute scaling assumptions
compute_doubling_months = 6  # Compute doubles every 6 months
models_per_year = 4  # Frontier models per year

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
    
    # Kaplan compute needed (theoretical baseline)
    kaplan_compute_needed = find_kaplan_compute(target_eci)
    print(f"  Kaplan baseline: {kaplan_compute_needed:.2e} FLOPs")
    
    # Tax at target time
    num_models_ahead = target_years * models_per_year
    future_model_idx = len(lucr_eci) + num_models_ahead
    future_tax = asymptotic_tax(future_model_idx, initial_tax_fit, decay_rate_fit)
    
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
    
    # If not achievable, how fast must compute double?
    if not achievable:
        # Find doubling period needed
        needed_multiplier = actual_compute_needed / current_compute
        doublings_needed = np.log2(needed_multiplier)
        months_needed = target_years * 12
        doubling_months_needed = months_needed / doublings_needed
        
        print(f"  Would need: {doubling_months_needed:.1f}-month doubling (not {compute_doubling_months} months)")
    
    # If achievable, when exactly?
    if achievable:
        # Binary search for exact timeline
        for test_years in np.linspace(0.1, target_years, 1000):
            test_months = test_years * 12
            test_compute_mult = 2 ** (test_months / compute_doubling_months)
            test_available = current_compute * test_compute_mult
            
            test_models_ahead = test_years * models_per_year
            test_tax = asymptotic_tax(len(lucr_eci) + test_models_ahead, initial_tax_fit, decay_rate_fit)
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
        'achievable': achievable
    })

forecast_df = pd.DataFrame(forecast_results)

# Save
forecast_df.to_csv('outputs/scenario_agi_forecasts.csv', index=False)
print(f"\nSaved to outputs/scenario_agi_forecasts.csv")

# Summary table
print(f"\n{'='*100}")
print("SUMMARY TABLE:")
print(f"{'='*100}")
print(forecast_df[['scenario', 'target_years', 'target_eci', 'gap', 'achievable']].to_string(index=False))

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Compute requirements vs availability
ax1 = axes[0, 0]

scenarios_list = forecast_df['scenario'].values
x_pos = np.arange(len(scenarios_list))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, forecast_df['available_compute'], width,
               label='Available (6mo doubling)', color='#06A77D', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x_pos + width/2, forecast_df['actual_compute_needed'], width,
               label='Needed (Kaplan + tax)', color='#E63946', alpha=0.7, edgecolor='black', linewidth=1.5)

ax1.set_yscale('log')
ax1.set_ylabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax1.set_title('Compute Requirements by Scenario', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(scenarios_list, rotation=45, ha='right')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which='both', axis='y')

# Plot 2: Gap ratio
ax2 = axes[0, 1]

colors = ['#06A77D' if achievable else '#E63946' for achievable in forecast_df['achievable']]
bars = ax2.bar(scenarios_list, forecast_df['gap'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

ax2.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Achievable threshold')
ax2.set_ylabel('Gap (Needed/Available)', fontsize=12, fontweight='bold')
ax2.set_title('Feasibility: Gap < 1 = Achievable', fontsize=14, fontweight='bold', pad=15)
ax2.set_yscale('log')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, which='both', axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: ECI targets
ax3 = axes[1, 0]

ax3.bar(scenarios_list, forecast_df['target_eci'], color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.axhline(current_eci, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'Current ({current_eci:.0f})')

max_eci_kaplan = a_linear * L_inf + b_linear
ax3.axhline(max_eci_kaplan, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Kaplan ceiling ({max_eci_kaplan:.0f})')

ax3.set_ylabel('Target ECI Score', fontsize=12, fontweight='bold')
ax3.set_title('ECI Requirements', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 4: Tax evolution
ax4 = axes[1, 1]

ax4.plot(forecast_df['target_years'], forecast_df['tax'], 'o-', linewidth=3, markersize=10, color='#F18F01')
ax4.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Perfect efficiency (1.0x)')

ax4.set_xlabel('Years', fontsize=12, fontweight='bold')
ax4.set_ylabel('Compute Tax', fontsize=12, fontweight='bold')
ax4.set_title('Tax Approaching Efficiency Over Time', fontsize=14, fontweight='bold', pad=15)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/scenario_agi_forecasts.png', dpi=300, bbox_inches='tight')
print("Saved visualization to outputs/scenario_agi_forecasts.png")

print(f"\n{'='*100}")
print("✅ SCENARIO FORECASTS COMPLETE!")
print(f"{'='*100}")

achievable_scenarios = forecast_df[forecast_df['achievable']]
if len(achievable_scenarios) > 0:
    print(f"\n🎯 ACHIEVABLE SCENARIOS:")
    for _, row in achievable_scenarios.iterrows():
        print(f"   • {row['scenario']:10s}: ECI {row['target_eci']:.0f} achievable with {compute_doubling_months}-month doubling")
else:
    print(f"\n⚠️  NO SCENARIOS ACHIEVABLE with {compute_doubling_months}-month compute doubling")
    print(f"   Compute scaling must accelerate OR algorithmic progress must diverge from Kaplan")


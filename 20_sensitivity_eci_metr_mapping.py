"""
Sensitivity Analysis: ECI → METR Mapping
Test different mappings to see how AGI timeline changes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

print("="*100)
print("SENSITIVITY ANALYSIS: ECI → METR MAPPING")
print("="*100)

# Load baseline mapping
with open('outputs/eci_to_metr_mapping.json', 'r') as f:
    baseline_mapping = json.load(f)

baseline_slope = baseline_mapping['slope']
baseline_intercept = baseline_mapping['intercept']
baseline_r2 = baseline_mapping['r_squared']

print(f"\nBaseline mapping (from {baseline_mapping['n_models']} models):")
print(f"  METR = {baseline_slope:.6f} * ECI + {baseline_intercept:.6f}")
print(f"  R² = {baseline_r2:.4f}")

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

# Load tax model
lucr_eci = pd.read_csv('outputs/lucr_eci_scale.csv')
lucr_eci = lucr_eci[lucr_eci['lucr_compute_tax'] < 10].copy()

# Fit asymptotic tax
def asymptotic_tax(t, initial_tax, decay_rate):
    return 1.0 + (initial_tax - 1.0) * np.exp(-decay_rate * t)

from scipy.optimize import curve_fit
time_idx = np.arange(len(lucr_eci))
tax_values = lucr_eci['lucr_compute_tax'].values
popt, _ = curve_fit(asymptotic_tax, time_idx, tax_values, p0=[tax_values[0], 0.01], maxfev=10000)
initial_tax_fit, decay_rate_fit = popt

# Function to find compute for target ECI
def find_kaplan_compute(target_eci):
    compute_range = np.logspace(20, 60, 100000)  # Extended range
    eci_range = np.array([kaplan_eci(c) for c in compute_range])
    idx = np.argmin(np.abs(eci_range - target_eci))
    
    # Check if we actually reached the target
    best_eci = eci_range[idx]
    if abs(best_eci - target_eci) > 50:  # More than 50 ECI points away
        print(f"  ⚠️  WARNING: Could only reach ECI {best_eci:.1f}, not {target_eci:.1f}")
    
    return compute_range[idx]

# Current state
current_compute = 1.51e25
compute_doubling_months = 6
models_per_year = 4

# Test different ECI→METR mappings
print("\n" + "="*100)
print("TESTING ALTERNATIVE MAPPINGS:")
print("="*100)

scenarios = []

# Scenario 1: Baseline (current fit)
scenarios.append({
    'name': 'Baseline (current fit)',
    'slope': baseline_slope,
    'intercept': baseline_intercept,
    'description': 'Linear fit from 5-7 overlapping models'
})

# Scenario 2: Steeper slope (2x) - METR is "easier" than we think
scenarios.append({
    'name': 'Optimistic (2x slope)',
    'slope': baseline_slope * 2,
    'intercept': baseline_intercept,
    'description': 'If METR scales faster with ECI'
})

# Scenario 3: Shallower slope (0.5x) - METR is "harder"
scenarios.append({
    'name': 'Pessimistic (0.5x slope)',
    'slope': baseline_slope * 0.5,
    'intercept': baseline_intercept,
    'description': 'If METR scales slower with ECI'
})

# Scenario 4: Different intercept (higher baseline)
scenarios.append({
    'name': 'Higher baseline (+0.1)',
    'slope': baseline_slope,
    'intercept': baseline_intercept + 0.1,
    'description': 'If current METR is underestimated'
})

# Scenario 5: Power law (METR = a * ECI^b)
# Fit power law to get comparable parameters
# For simplicity, approximate: METR ≈ 0.0001 * ECI^1.5
scenarios.append({
    'name': 'Power law (superlinear)',
    'slope': None,  # Special case
    'intercept': None,
    'power_a': 0.00001,
    'power_b': 2.0,
    'description': 'If METR grows superlinearly with ECI'
})

results = []

for scenario in scenarios:
    print(f"\n{'='*100}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"{'='*100}")
    
    # Define mapping for this scenario
    if scenario['slope'] is not None:
        # Linear
        def eci_to_metr(eci):
            return scenario['slope'] * eci + scenario['intercept']
    else:
        # Power law
        def eci_to_metr(eci):
            return scenario['power_a'] * (eci ** scenario['power_b'])
    
    # Target METR = 0.9
    target_metr = 0.9
    
    # Find required ECI (solve numerically)
    if scenario['slope'] is not None:
        target_eci = (target_metr - scenario['intercept']) / scenario['slope']
    else:
        # For power law: ECI = (METR / a)^(1/b)
        target_eci = (target_metr / scenario['power_a']) ** (1.0 / scenario['power_b'])
    
    print(f"\nTarget: METR = {target_metr:.2f} → ECI = {target_eci:.1f}")
    
    # Kaplan compute needed
    kaplan_compute_needed = find_kaplan_compute(target_eci)
    print(f"Kaplan baseline: {kaplan_compute_needed:.2e} FLOPs")
    
    # Find when we can achieve it
    agi_achieved = False
    agi_years = None
    
    for years in range(1, 101):  # Check up to 100 years
        months = years * 12
        compute_multiplier = 2 ** (months / compute_doubling_months)
        available_compute = current_compute * compute_multiplier
        
        # Tax at this time
        num_models_ahead = years * models_per_year
        future_model_idx = len(lucr_eci) + num_models_ahead
        future_tax = asymptotic_tax(future_model_idx, initial_tax_fit, decay_rate_fit)
        
        # Actual compute needed
        actual_compute_needed = kaplan_compute_needed * future_tax
        
        if available_compute >= actual_compute_needed:
            agi_achieved = True
            agi_years = years
            agi_compute = available_compute
            agi_tax = future_tax
            break
    
    if agi_achieved:
        print(f"\n✓ AGI ACHIEVED in ~{agi_years} years (2025 + {agi_years} = {2025 + agi_years})")
        print(f"  Compute: {agi_compute:.2e} FLOPs")
        print(f"  Tax: {agi_tax:.3f}x")
    else:
        print(f"\n✗ AGI NOT achieved within 100 years")
        print(f"  Need: {kaplan_compute_needed:.2e} FLOPs")
    
    results.append({
        'scenario': scenario['name'],
        'target_eci': target_eci,
        'kaplan_compute': kaplan_compute_needed,
        'agi_years': agi_years if agi_achieved else np.inf,
        'achieved': agi_achieved
    })

results_df = pd.DataFrame(results)

# Summary comparison
print("\n" + "="*100)
print("TIMELINE COMPARISON:")
print("="*100)

results_df_sorted = results_df.sort_values('agi_years')

for _, row in results_df_sorted.iterrows():
    if row['achieved']:
        print(f"\n{row['scenario']:30s} → {row['agi_years']:3.0f} years (ECI target: {row['target_eci']:6.1f})")
    else:
        print(f"\n{row['scenario']:30s} → >100 years (ECI target: {row['target_eci']:6.1f})")

# Save results
results_df.to_csv('outputs/eci_metr_sensitivity.csv', index=False)
print(f"\n\nSaved sensitivity analysis to outputs/eci_metr_sensitivity.csv")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Timeline comparison
ax1 = axes[0, 0]

achieved_results = results_df[results_df['achieved']].copy()
if len(achieved_results) > 0:
    colors = ['#06A77D' if r == results_df.loc[0, 'agi_years'] else '#2E86AB' for r in achieved_results['agi_years']]
    bars = ax1.barh(range(len(achieved_results)), achieved_results['agi_years'], 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax1.set_yticks(range(len(achieved_results)))
    ax1.set_yticklabels(achieved_results['scenario'])
    ax1.set_xlabel('Years to AGI', fontsize=12, fontweight='bold')
    ax1.set_title('AGI Timeline Sensitivity', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Annotate bars
    for i, years in enumerate(achieved_results['agi_years']):
        ax1.text(years + 1, i, f'{years:.0f}y ({2025 + years:.0f})', 
                va='center', fontsize=10, fontweight='bold')

# Plot 2: ECI targets
ax2 = axes[0, 1]

colors_eci = ['#E63946' if eci > 1000 else '#F18F01' if eci > 500 else '#06A77D' for eci in results_df['target_eci']]
bars = ax2.barh(range(len(results_df)), results_df['target_eci'],
               color=colors_eci, alpha=0.7, edgecolor='black', linewidth=2)

ax2.set_yticks(range(len(results_df)))
ax2.set_yticklabels(results_df['scenario'])
ax2.set_xlabel('Target ECI for METR = 0.9', fontsize=12, fontweight='bold')
ax2.set_title('ECI Requirements by Scenario', fontsize=14, fontweight='bold', pad=15)
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3, which='both', axis='x')

# Mark baseline
baseline_eci = results_df.iloc[0]['target_eci']
ax2.axvline(baseline_eci, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
ax2.legend(fontsize=10)

# Plot 3: Different ECI→METR curves
ax3 = axes[1, 0]

eci_range = np.linspace(100, 200, 200)

for scenario in scenarios[:4]:  # Skip power law for clarity
    if scenario['slope'] is not None:
        metr_range = scenario['slope'] * eci_range + scenario['intercept']
        metr_range = np.clip(metr_range, 0, 1.1)  # Clip to reasonable range
        
        linestyle = '-' if scenario['name'] == 'Baseline (current fit)' else '--'
        linewidth = 3 if scenario['name'] == 'Baseline (current fit)' else 2
        
        ax3.plot(eci_range, metr_range * 100, linestyle, linewidth=linewidth,
                label=scenario['name'], alpha=0.8)

ax3.axhline(90, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='AGI Target (90%)')
ax3.set_xlabel('ECI Score', fontsize=12, fontweight='bold')
ax3.set_ylabel('METR Frontier Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('Alternative ECI → METR Mappings', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 100])

# Plot 4: Uncertainty bands
ax4 = axes[1, 1]

# Show range of timelines
if len(achieved_results) > 0:
    min_years = achieved_results['agi_years'].min()
    max_years = achieved_results['agi_years'].max()
    mean_years = achieved_results['agi_years'].mean()
    
    ax4.barh([0], [mean_years], height=0.5, color='#2E86AB', alpha=0.7, 
            edgecolor='black', linewidth=2, label='Mean')
    ax4.barh([0], [max_years - min_years], left=[min_years], height=0.5,
            color='#F18F01', alpha=0.3, label='Range')
    
    ax4.set_yticks([])
    ax4.set_xlabel('Years to AGI', fontsize=12, fontweight='bold')
    ax4.set_title('Timeline Uncertainty', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.legend(fontsize=10)
    
    # Add text
    ax4.text(mean_years, -0.15, f'Mean: {mean_years:.0f}y', ha='center', fontsize=11, fontweight='bold')
    ax4.text(min_years, 0.3, f'{min_years:.0f}y', ha='center', fontsize=10)
    ax4.text(max_years, 0.3, f'{max_years:.0f}y', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/eci_metr_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
print("Saved visualization to outputs/eci_metr_sensitivity_analysis.png")

print("\n" + "="*100)
print("✅ SENSITIVITY ANALYSIS COMPLETE!")
print("="*100)

print(f"\n🎯 KEY FINDINGS:")
print(f"   Timeline range: {achieved_results['agi_years'].min():.0f} - {achieved_results['agi_years'].max():.0f} years")
print(f"   Baseline (current fit): {results_df.iloc[0]['agi_years']:.0f} years")
print(f"   Most optimistic scenario: {achieved_results.iloc[0]['scenario']} ({achieved_results.iloc[0]['agi_years']:.0f} years)")
print(f"\n   The ECI→METR mapping is CRITICAL to the forecast.")
print(f"   With only {baseline_mapping['n_models']} data points, there's significant uncertainty.")


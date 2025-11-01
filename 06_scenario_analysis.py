"""
STEP 6: Comprehensive Scenario Analysis

Creates detailed visualizations comparing:
- Acceleration scenarios (AGI race)
- Baseline
- Efficiency improvements
- Slowdown/restrictions

Focus on uncertainty and policy-relevant insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*100)
print("STEP 6: COMPREHENSIVE SCENARIO ANALYSIS")
print("="*100)

# Load results
timelines = pd.read_csv('outputs/scenario_timelines.csv')

print(f"\nLoaded {len(timelines)} scenario results")

# ============================================================================
# Organize scenarios by type
# ============================================================================

scenario_groups = {
    'Acceleration (AGI Race)': [
        'AGI Race (3mo doubling)',
        'Moderate Acceleration (4mo)',
        'AGI Race + 20% Efficiency'
    ],
    'Baseline': [
        'Baseline (Current Trends)'
    ],
    'Efficiency Improvements': [
        '+5% Translation',
        '+10% Translation',
        '+20% Translation',
        '+30% Translation'
    ],
    'Slowdown/Restrictions': [
        'Compute Restrictions (12mo)',
        'Compute Restrictions (18mo)',
        '+20% + Slower (9mo)',
        'Faster LUCR Decay (90% β)'
    ]
}

# ============================================================================
# Plot 1: Timeline comparison (all scenarios)
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle('AGI Timeline Scenarios: Comprehensive Analysis', fontsize=18, fontweight='bold')

# Near-AGI (0.8)
ax = axes[0]

near_agi = timelines[timelines['threshold'] == 'Near-AGI'].copy()
near_agi = near_agi.sort_values('median_years')

y_pos = np.arange(len(near_agi))
colors = []
for scenario in near_agi['scenario']:
    if any(scenario in group for group in scenario_groups['Acceleration (AGI Race)']):
        colors.append('purple')
    elif any(scenario in group for group in scenario_groups['Efficiency Improvements']):
        colors.append('green')
    elif any(scenario in group for group in scenario_groups['Slowdown/Restrictions']):
        colors.append('red')
    else:
        colors.append('gray')

ax.barh(y_pos, near_agi['median_years'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Error bars
for i, row in enumerate(near_agi.itertuples()):
    if not pd.isna(row.ci_low_years):
        ax.plot([row.ci_low_years, row.ci_high_years], [i, i], 'k-', linewidth=2, alpha=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(near_agi['scenario'], fontsize=10)
ax.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax.set_title('Near-AGI (METR 0.8) Timeline', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Add probability labels
for i, row in enumerate(near_agi.itertuples()):
    label = f"{row.probability*100:.0f}%"
    if row.probability > 0:
        ax.text(row.median_years + 0.3, i, label, va='center', fontsize=9, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='purple', alpha=0.7, edgecolor='black', label='Acceleration (AGI Race)'),
    Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='Baseline'),
    Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Efficiency Improvements'),
    Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Slowdown/Restrictions')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# AGI (0.9)
ax = axes[1]

agi = timelines[timelines['threshold'] == 'AGI'].copy()
agi = agi.sort_values('median_years')

y_pos = np.arange(len(agi))
colors = []
for scenario in agi['scenario']:
    if any(scenario in group for group in scenario_groups['Acceleration (AGI Race)']):
        colors.append('purple')
    elif any(scenario in group for group in scenario_groups['Efficiency Improvements']):
        colors.append('green')
    elif any(scenario in group for group in scenario_groups['Slowdown/Restrictions']):
        colors.append('red')
    else:
        colors.append('gray')

ax.barh(y_pos, agi['median_years'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Error bars
for i, row in enumerate(agi.itertuples()):
    if not pd.isna(row.ci_low_years):
        ax.plot([row.ci_low_years, row.ci_high_years], [i, i], 'k-', linewidth=2, alpha=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(agi['scenario'], fontsize=10)
ax.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax.set_title('AGI (METR 0.9) Timeline', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Add probability labels
for i, row in enumerate(agi.itertuples()):
    label = f"{row.probability*100:.0f}%" if row.probability > 0 else "0%"
    if row.probability > 0:
        ax.text(row.median_years + 0.3, i, label, va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/03_scenario_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: outputs/03_scenario_analysis_comprehensive.png")

# ============================================================================
# Plot 2: Key scenarios comparison (focused)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Key Scenario Comparison', fontsize=18, fontweight='bold')

key_scenarios = [
    'AGI Race (3mo doubling)',
    'Baseline (Current Trends)',
    '+20% Translation',
    'Faster LUCR Decay (90% β)',
    'Compute Restrictions (12mo)',
    'AGI Race + 20% Efficiency'
]

# Near-AGI
ax = axes[0]
near_agi_key = near_agi[near_agi['scenario'].isin(key_scenarios)].copy()
# Sort by key_scenarios order
near_agi_key['scenario_order'] = near_agi_key['scenario'].apply(lambda x: key_scenarios.index(x))
near_agi_key = near_agi_key.sort_values('scenario_order')

y_pos = np.arange(len(near_agi_key))
colors_key = ['purple', 'gray', 'green', 'brown', 'red', 'darkviolet']

bars = ax.barh(y_pos, near_agi_key['median_years'], color=colors_key, alpha=0.7, 
               edgecolor='black', linewidth=2)

# Error bars
for i, row in enumerate(near_agi_key.itertuples()):
    if not pd.isna(row.ci_low_years):
        ax.plot([row.ci_low_years, row.ci_high_years], [i, i], 'k-', linewidth=3, alpha=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(near_agi_key['scenario'], fontsize=11)
ax.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax.set_title('Near-AGI (0.8)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Add median year labels
for i, row in enumerate(near_agi_key.itertuples()):
    ax.text(row.median_years + 0.2, i, f'{row.median_years:.1f}y', 
            va='center', fontsize=10, fontweight='bold')

# AGI
ax = axes[1]
agi_key = agi[agi['scenario'].isin(key_scenarios)].copy()
# Sort by key_scenarios order
agi_key['scenario_order'] = agi_key['scenario'].apply(lambda x: key_scenarios.index(x))
agi_key = agi_key.sort_values('scenario_order')

y_pos = np.arange(len(agi_key))

bars = ax.barh(y_pos, agi_key['median_years'], color=colors_key, alpha=0.7, 
               edgecolor='black', linewidth=2)

# Error bars
for i, row in enumerate(agi_key.itertuples()):
    if not pd.isna(row.ci_low_years):
        ax.plot([row.ci_low_years, row.ci_high_years], [i, i], 'k-', linewidth=3, alpha=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(agi_key['scenario'], fontsize=11)
ax.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax.set_title('AGI (0.9)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Add median year labels
for i, row in enumerate(agi_key.itertuples()):
    ax.text(row.median_years + 0.2, i, f'{row.median_years:.1f}y', 
            va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/04_key_scenarios.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: outputs/04_key_scenarios.png")

# ============================================================================
# Plot 3: Probability analysis
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Probability of Achieving Thresholds (within 10 years)', fontsize=18, fontweight='bold')

# By scenario type
ax = axes[0]

scenario_type_probs = []
for group_name, scenarios in scenario_groups.items():
    near_prob = timelines[(timelines['scenario'].isin(scenarios)) & 
                          (timelines['threshold'] == 'Near-AGI')]['probability'].mean()
    agi_prob = timelines[(timelines['scenario'].isin(scenarios)) & 
                         (timelines['threshold'] == 'AGI')]['probability'].mean()
    scenario_type_probs.append({
        'type': group_name,
        'Near-AGI': near_prob,
        'AGI': agi_prob
    })

prob_df = pd.DataFrame(scenario_type_probs)
x_pos = np.arange(len(prob_df))
width = 0.35

ax.bar(x_pos - width/2, prob_df['Near-AGI']*100, width, 
       label='Near-AGI (0.8)', color='orange', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.bar(x_pos + width/2, prob_df['AGI']*100, width, 
       label='AGI (0.9)', color='red', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(prob_df['type'], fontsize=11, rotation=15, ha='right')
ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
ax.set_title('Average Probability by Scenario Type', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 105])

# Add value labels
for i, row in prob_df.iterrows():
    ax.text(i - width/2, row['Near-AGI']*100 + 2, f"{row['Near-AGI']*100:.0f}%",
            ha='center', fontsize=9, fontweight='bold')
    ax.text(i + width/2, row['AGI']*100 + 2, f"{row['AGI']*100:.0f}%",
            ha='center', fontsize=9, fontweight='bold')

# Key scenarios
ax = axes[1]

prob_key = timelines[timelines['scenario'].isin(key_scenarios)].copy()
near_probs = prob_key[prob_key['threshold'] == 'Near-AGI']['probability'].values * 100
agi_probs = prob_key[prob_key['threshold'] == 'AGI']['probability'].values * 100

x_pos = np.arange(len(key_scenarios))
width = 0.35

ax.bar(x_pos - width/2, near_probs, width, 
       label='Near-AGI (0.8)', color='orange', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.bar(x_pos + width/2, agi_probs, width, 
       label='AGI (0.9)', color='red', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(key_scenarios, fontsize=10, rotation=20, ha='right')
ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
ax.set_title('Probability for Key Scenarios', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 105])

plt.tight_layout()
plt.savefig('outputs/05_probability_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: outputs/05_probability_analysis.png")

# ============================================================================
# Summary statistics
# ============================================================================

print(f"\n{'='*100}")
print("SCENARIO ANALYSIS SUMMARY")
print("="*100)

print(f"\n📊 AGI RACE SCENARIOS (Acceleration):")
for scenario in scenario_groups['Acceleration (AGI Race)']:
    near = timelines[(timelines['scenario']==scenario) & (timelines['threshold']=='Near-AGI')]
    agi = timelines[(timelines['scenario']==scenario) & (timelines['threshold']=='AGI')]
    if len(near) > 0 and len(agi) > 0:
        print(f"\n  {scenario}:")
        print(f"    Near-AGI: {near['median_date'].values[0]} ({near['median_years'].values[0]:.1f}y, {near['probability'].values[0]*100:.0f}%)")
        print(f"    AGI: {agi['median_date'].values[0]} ({agi['median_years'].values[0]:.1f}y, {agi['probability'].values[0]*100:.0f}%)")

print(f"\n📊 BASELINE:")
baseline_near = timelines[(timelines['scenario']=='Baseline (Current Trends)') & (timelines['threshold']=='Near-AGI')]
baseline_agi = timelines[(timelines['scenario']=='Baseline (Current Trends)') & (timelines['threshold']=='AGI')]
print(f"  Near-AGI: {baseline_near['median_date'].values[0]} ({baseline_near['median_years'].values[0]:.1f}y, {baseline_near['probability'].values[0]*100:.0f}%)")
print(f"  AGI: {baseline_agi['median_date'].values[0]} ({baseline_agi['median_years'].values[0]:.1f}y, {baseline_agi['probability'].values[0]*100:.0f}%)")

print(f"\n📊 SLOWDOWN/RESTRICTIONS:")
for scenario in scenario_groups['Slowdown/Restrictions']:
    near = timelines[(timelines['scenario']==scenario) & (timelines['threshold']=='Near-AGI')]
    agi = timelines[(timelines['scenario']==scenario) & (timelines['threshold']=='AGI')]
    if len(near) > 0:
        near_str = f"{near['median_date'].values[0]} ({near['median_years'].values[0]:.1f}y, {near['probability'].values[0]*100:.0f}%)" if not pd.isna(near['median_years'].values[0]) else "Not achieved"
        agi_str = f"{agi['median_date'].values[0]} ({agi['median_years'].values[0]:.1f}y, {agi['probability'].values[0]*100:.0f}%)" if not pd.isna(agi['median_years'].values[0]) and agi['probability'].values[0] > 0 else "Not achieved"
        print(f"\n  {scenario}:")
        print(f"    Near-AGI: {near_str}")
        print(f"    AGI: {agi_str}")

print(f"\n📊 KEY INSIGHTS:")
race_3mo_agi = timelines[(timelines['scenario']=='AGI Race (3mo doubling)') & (timelines['threshold']=='AGI')]['median_years'].values[0]
baseline_agi_years = baseline_agi['median_years'].values[0]
slow_12mo_agi = timelines[(timelines['scenario']=='Compute Restrictions (12mo)') & (timelines['threshold']=='AGI')]['median_years'].values[0]

print(f"  • AGI Race (3mo): AGI in {race_3mo_agi:.1f} years vs Baseline {baseline_agi_years:.1f} years = {baseline_agi_years - race_3mo_agi:.1f} years faster")
print(f"  • Restrictions (12mo): AGI in {slow_12mo_agi:.1f} years vs Baseline {baseline_agi_years:.1f} years = {slow_12mo_agi - baseline_agi_years:.1f} years slower")
print(f"  • Range: {race_3mo_agi:.1f} - {slow_12mo_agi:.1f} years ({slow_12mo_agi/race_3mo_agi:.1f}x difference)")

# Most extreme scenario
race_eff = timelines[(timelines['scenario']=='AGI Race + 20% Efficiency') & (timelines['threshold']=='AGI')]['median_years'].values[0]
print(f"  • Most aggressive (Race + Efficiency): AGI in {race_eff:.1f} years ({timelines[(timelines['scenario']=='AGI Race + 20% Efficiency') & (timelines['threshold']=='AGI')]['median_date'].values[0]})")

print(f"\n{'='*100}")
print("✅ STEP 6 COMPLETE")
print("="*100)

print(f"\n💡 All scenario visualizations saved to outputs/")
print(f"  • 03_scenario_analysis_comprehensive.png")
print(f"  • 04_key_scenarios.png")
print(f"  • 05_probability_analysis.png")


"""
STEP 5: Generate Visualizations

Creates all plots for the competition:
1. Data overview (METR filtering, task distribution)
2. Model fits (C→ECI, ECI→METR)
3. LUCR decomposition
4. Scenario timelines
5. Summary dashboard

Output:
- Multiple PNG files in outputs/
- One comprehensive PDF report
"""

import pandas as pd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

print("="*100)
print("STEP 5: GENERATE VISUALIZATIONS")
print("="*100)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# Load all data
# ============================================================================

# Data
stage1 = pd.read_csv('outputs/stage1_compute_eci.csv')
stage2 = pd.read_csv('outputs/stage2_eci_metr.csv')
current = pd.read_csv('outputs/current_state.csv')
metr_cap = pd.read_csv('outputs/metr_capability_summary.csv')

# Model
posterior = pd.read_csv('outputs/posterior_samples.csv')
summary = pd.read_csv('outputs/posterior_summary.csv')

# Results
timelines = pd.read_csv('outputs/scenario_timelines.csv')

print(f"\nLoaded data:")
print(f"  Stage 1 (C→ECI): {len(stage1)} models")
print(f"  Stage 2 (ECI→METR): {len(stage2)} models")
print(f"  Posterior samples: {len(posterior)}")
print(f"  Scenario results: {len(timelines)}")

# ============================================================================
# Plot 1: Data Overview
# ============================================================================

print(f"\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('AGI Forecast: Data Overview', fontsize=18, fontweight='bold')

# 1a: METR capability vs compute
ax = axes[0, 0]
ax.scatter(metr_cap['compute'], metr_cap['metr_capability'], 
           s=150, alpha=0.7, c=range(len(metr_cap)), cmap='viridis',
           edgecolors='black', linewidth=1.5)

for _, row in metr_cap.iterrows():
    ax.text(row['compute']*1.1, row['metr_capability'], row['model'],
            fontsize=8, va='center', alpha=0.8)

ax.set_xscale('log')
ax.set_xlabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax.set_ylabel('METR Capability (Subhuman Tasks)', fontsize=12, fontweight='bold')
ax.set_title('Method B: Run-Level Filtering (Model < Human)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# 1b: Stage 1 (C→ECI)
ax = axes[0, 1]
ax.scatter(stage1['log_compute'], stage1['eci'], 
           s=100, alpha=0.6, c='steelblue', edgecolors='black')

# Plot posterior predictive
compute_range = np.logspace(20, 28, 100)
log_compute_range = np.log10(compute_range)

# Sample 100 posterior trajectories
for i in np.random.choice(len(posterior), 100):
    eci_pred = posterior['a'].iloc[i] + posterior['b'].iloc[i] * log_compute_range
    ax.plot(log_compute_range, eci_pred, 'b-', alpha=0.02)

# Median
eci_median = posterior['a'].mean() + posterior['b'].mean() * log_compute_range
ax.plot(log_compute_range, eci_median, 'r-', linewidth=3, label='Posterior Median')

ax.set_xlabel('log₁₀(Training Compute)', fontsize=12, fontweight='bold')
ax.set_ylabel('ECI Score', fontsize=12, fontweight='bold')
ax.set_title('Stage 1: Compute → ECI (Epoch Data)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 1c: Stage 2 (ECI→METR)
ax = axes[1, 0]
ax.scatter(stage2['eci'], stage2['metr'], 
           s=200, alpha=0.7, c='orange', edgecolors='black', linewidth=2)

for _, row in stage2.iterrows():
    ax.text(row['eci'], row['metr']+0.01, row['metr_model'],
            fontsize=8, ha='center', va='bottom', alpha=0.8)

# Plot posterior predictive
eci_range = np.linspace(stage2['eci'].min()-10, stage2['eci'].max()+30, 100)

for i in np.random.choice(len(posterior), 100):
    logit_metr_pred = posterior['alpha'].iloc[i] + posterior['beta'].iloc[i] * eci_range
    metr_pred = expit(logit_metr_pred)
    ax.plot(eci_range, metr_pred, 'orange', alpha=0.02)

# Median
logit_metr_median = posterior['alpha'].mean() + posterior['beta'].mean() * eci_range
metr_median = expit(logit_metr_median)
ax.plot(eci_range, metr_median, 'r-', linewidth=3, label='Posterior Median')

ax.set_xlabel('ECI Score', fontsize=12, fontweight='bold')
ax.set_ylabel('METR Capability', fontsize=12, fontweight='bold')
ax.set_title('Stage 2: ECI → METR (Overlap Models)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 1d: LUCR decomposition
ax = axes[1, 1]

lucr_data = summary[summary['parameter'].isin(['scaling_eff', 'translation_eff', 'lucr'])]
params = lucr_data['parameter'].values
means = lucr_data['mean'].values
stds = lucr_data['std'].values

x_pos = np.arange(len(params))
ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7,
       color=['steelblue', 'orange', 'green'],
       edgecolor='black', linewidth=1.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(['Scaling\n(dECI/d log C)', 'Translation\n(dMETR/dECI)', 'Combined\n(dMETR/d log C)'],
                   fontsize=11)
ax.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
ax.set_title('LUCR Decomposition', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.01, f'{mean:.3f}\n±{std:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/01_data_overview.png', dpi=300, bbox_inches='tight')
print("  ✅ Saved: outputs/01_data_overview.png")

# ============================================================================
# Plot 2: Scenario Timelines
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('AGI Timeline Scenarios', fontsize=18, fontweight='bold')

# 2a: Baseline timeline probabilities
ax = axes[0, 0]

baseline_near = timelines[(timelines['scenario']=='Baseline (Current Trends)') & 
                          (timelines['threshold']=='Near-AGI')]
baseline_agi = timelines[(timelines['scenario']=='Baseline (Current Trends)') & 
                         (timelines['threshold']=='AGI')]

data = [
    {'threshold': 'Near-AGI\n(0.8)', 
     'median': baseline_near['median_years'].values[0],
     'low': baseline_near['ci_low_years'].values[0],
     'high': baseline_near['ci_high_years'].values[0],
     'prob': baseline_near['probability'].values[0]},
    {'threshold': 'AGI\n(0.9)', 
     'median': baseline_agi['median_years'].values[0],
     'low': baseline_agi['ci_low_years'].values[0],
     'high': baseline_agi['ci_high_years'].values[0],
     'prob': baseline_agi['probability'].values[0]}
]

x_pos = [0, 1]
medians = [d['median'] for d in data]
lows = [d['low'] for d in data]
highs = [d['high'] for d in data]
probs = [d['prob'] for d in data]

bars = ax.bar(x_pos, medians, color=['orange', 'red'], alpha=0.7,
              edgecolor='black', linewidth=2)

# Error bars
for i, (low, high, median) in enumerate(zip(lows, highs, medians)):
    ax.plot([i, i], [low, high], 'k-', linewidth=3, alpha=0.5)
    ax.plot([i-0.1, i+0.1], [low, low], 'k-', linewidth=2, alpha=0.5)
    ax.plot([i-0.1, i+0.1], [high, high], 'k-', linewidth=2, alpha=0.5)

# Labels
ax.set_xticks(x_pos)
ax.set_xticklabels([d['threshold'] for d in data], fontsize=12)
ax.set_ylabel('Years from Now', fontsize=12, fontweight='bold')
ax.set_title('Baseline (6-month compute doubling)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add probability labels
for i, (median, prob) in enumerate(zip(medians, probs)):
    ax.text(i, median+0.3, f'{prob*100:.0f}% prob\n{median:.1f} years',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2b: Efficiency scenarios (Near-AGI)
ax = axes[0, 1]

eff_scenarios = ['Baseline (Current Trends)', '+5% Translation', '+10% Translation',
                 '+20% Translation', '+30% Translation']
eff_data = []

for scenario in eff_scenarios:
    row = timelines[(timelines['scenario']==scenario) & (timelines['threshold']=='Near-AGI')]
    if len(row) > 0 and row['probability'].values[0] > 0:
        eff_data.append({
            'scenario': scenario.replace(' (Current Trends)', '').replace(' Translation', ''),
            'median': row['median_years'].values[0],
            'low': row['ci_low_years'].values[0],
            'high': row['ci_high_years'].values[0],
            'prob': row['probability'].values[0]
        })

x_pos = np.arange(len(eff_data))
medians = [d['median'] for d in eff_data]
lows = [d['low'] for d in eff_data]
highs = [d['high'] for d in eff_data]

ax.barh(x_pos, medians, color=['gray', 'lightgreen', 'mediumseagreen', 'green', 'darkgreen'],
        alpha=0.7, edgecolor='black', linewidth=1.5)

# Error bars
for i, (low, high) in enumerate(zip(lows, highs)):
    ax.plot([low, high], [i, i], 'k-', linewidth=2, alpha=0.5)

ax.set_yticks(x_pos)
ax.set_yticklabels([d['scenario'] for d in eff_data], fontsize=10)
ax.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax.set_title('Near-AGI (0.8): Efficiency Scenarios', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# 2c: Efficiency scenarios (AGI)
ax = axes[1, 0]

agi_data = []
for scenario in eff_scenarios:
    row = timelines[(timelines['scenario']==scenario) & (timelines['threshold']=='AGI')]
    if len(row) > 0 and row['probability'].values[0] > 0:
        agi_data.append({
            'scenario': scenario.replace(' (Current Trends)', '').replace(' Translation', ''),
            'median': row['median_years'].values[0],
            'low': row['ci_low_years'].values[0],
            'high': row['ci_high_years'].values[0],
            'prob': row['probability'].values[0]
        })

x_pos = np.arange(len(agi_data))
medians = [d['median'] for d in agi_data]
lows = [d['low'] for d in agi_data]
highs = [d['high'] for d in agi_data]

ax.barh(x_pos, medians, color=['gray', 'lightgreen', 'mediumseagreen', 'green', 'darkgreen'],
        alpha=0.7, edgecolor='black', linewidth=1.5)

for i, (low, high) in enumerate(zip(lows, highs)):
    ax.plot([low, high], [i, i], 'k-', linewidth=2, alpha=0.5)

ax.set_yticks(x_pos)
ax.set_yticklabels([d['scenario'] for d in agi_data], fontsize=10)
ax.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax.set_title('AGI (0.9): Efficiency Scenarios', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# 2d: Probability comparison
ax = axes[1, 1]

scenarios_list = ['Baseline', '+5%', '+10%', '+20%', '+30%']
near_agi_probs = []
agi_probs = []

for short_name, full_name in zip(scenarios_list, eff_scenarios):
    near = timelines[(timelines['scenario']==full_name) & (timelines['threshold']=='Near-AGI')]
    agi = timelines[(timelines['scenario']==full_name) & (timelines['threshold']=='AGI')]
    
    near_agi_probs.append(near['probability'].values[0] if len(near) > 0 else 0)
    agi_probs.append(agi['probability'].values[0] if len(agi) > 0 else 0)

x_pos = np.arange(len(scenarios_list))
width = 0.35

ax.bar(x_pos - width/2, [p*100 for p in near_agi_probs], width,
       label='Near-AGI (0.8)', color='orange', alpha=0.7, edgecolor='black')
ax.bar(x_pos + width/2, [p*100 for p in agi_probs], width,
       label='AGI (0.9)', color='red', alpha=0.7, edgecolor='black')

ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios_list, fontsize=11)
ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
ax.set_title('Probability of Achieving Thresholds (within 10 years)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/02_scenario_timelines.png', dpi=300, bbox_inches='tight')
print("  ✅ Saved: outputs/02_scenario_timelines.png")

# ============================================================================
# Summary statistics
# ============================================================================

print(f"\n{'='*100}")
print("SUMMARY STATISTICS")
print("="*100)

best_current = current.iloc[0]
print(f"\n📊 Current State:")
print(f"  Model: {best_current['model']}")
print(f"  Compute: {best_current['compute']:.2e} FLOPs")
print(f"  METR (subhuman tasks): {best_current['metr_capability']:.3f}")

print(f"\n📊 Model Parameters:")
beta_mean = posterior['beta'].mean()
beta_std = posterior['beta'].std()
print(f"  Translation efficiency (β): {beta_mean:.4f} ± {beta_std:.4f}")

lucr_mean = summary[summary['parameter']=='lucr']['mean'].values[0]
lucr_std = summary[summary['parameter']=='lucr']['std'].values[0]
print(f"  Combined LUCR: {lucr_mean:.3f} ± {lucr_std:.3f}")

print(f"\n📊 Baseline Timeline:")
baseline_near_agi = timelines[(timelines['scenario']=='Baseline (Current Trends)') & (timelines['threshold']=='Near-AGI')]
baseline_agi = timelines[(timelines['scenario']=='Baseline (Current Trends)') & (timelines['threshold']=='AGI')]

print(f"  Near-AGI (0.8): {baseline_near_agi['probability'].values[0]*100:.0f}% prob, {baseline_near_agi['median_date'].values[0]}")
print(f"  AGI (0.9): {baseline_agi['probability'].values[0]*100:.0f}% prob, {baseline_agi['median_date'].values[0] if not pd.isna(baseline_agi['median_years'].values[0]) else 'Not achieved'}")

print(f"\n{'='*100}")
print("✅ STEP 5 COMPLETE")
print("="*100)

print(f"\n💡 All visualizations saved to outputs/")
print(f"  • 01_data_overview.png")
print(f"  • 02_scenario_timelines.png")


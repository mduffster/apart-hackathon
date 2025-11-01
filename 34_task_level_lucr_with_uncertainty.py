"""
Task-Level LUCR Analysis with Compute Uncertainty
Show which tasks benefit from algorithmic progress vs scaling limits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import json

print("="*100)
print("TASK-LEVEL LUCR ANALYSIS WITH UNCERTAINTY")
print("="*100)

# Load METR runs (granular task-level data)
runs = pd.read_json('data/METR/all_runs.jsonl', lines=True)

print(f"\nMETR data: {len(runs)} task runs")
print(f"  Models: {runs['model'].nunique()}")
print(f"  Task families: {runs['task_family'].nunique()}")

# Load compute data (known + estimated)
compute_data = {
    # Known (from ECI data)
    'claude_3_5_sonnet': {'compute': 2.70e25, 'source': 'known', 'conf_low': 2.70e25, 'conf_high': 2.70e25},
    'claude_3_5_sonnet_20241022': {'compute': 2.70e25, 'source': 'known', 'conf_low': 2.70e25, 'conf_high': 2.70e25},
    'gpt_4_0314': {'compute': 2.10e25, 'source': 'known', 'conf_low': 2.10e25, 'conf_high': 2.10e25},
    'gpt-4-0314': {'compute': 2.10e25, 'source': 'known', 'conf_low': 2.10e25, 'conf_high': 2.10e25},
    
    # Estimated (from ECI → Compute bridge)
    'gpt_4o': {'compute': 7.13e24, 'source': 'estimated', 'conf_low': 1.43e24, 'conf_high': 3.57e25},  # 5x range
    'o1_preview': {'compute': 2.59e25, 'source': 'estimated', 'conf_low': 5.18e24, 'conf_high': 1.30e26},  # 5x range
    'claude-3-7-sonnet-20250219': {'compute': 3.35e25, 'source': 'estimated', 'conf_low': 6.70e24, 'conf_high': 1.68e26},  # 5x range
}

print(f"\n{'='*100}")
print("COMPUTE DATA:")
print(f"{'='*100}")

for model, data in compute_data.items():
    if data['source'] == 'known':
        print(f"\n{model}: {data['compute']:.2e} FLOPs (known)")
    else:
        print(f"\n{model}: {data['compute']:.2e} FLOPs (estimated)")
        print(f"  Range: {data['conf_low']:.2e} - {data['conf_high']:.2e}")

# Match METR models to compute
runs['compute'] = runs['model'].map(lambda m: compute_data.get(m, {}).get('compute'))
runs['compute_source'] = runs['model'].map(lambda m: compute_data.get(m, {}).get('source', 'missing'))

runs_with_compute = runs[runs['compute'].notna()].copy()

print(f"\n{'='*100}")
print(f"MATCHED DATA:")
print(f"{'='*100}")
print(f"\nTask runs with compute: {len(runs_with_compute)}/{len(runs)}")
print(f"Models with compute: {runs_with_compute['model'].nunique()}")
print(f"Task families covered: {runs_with_compute['task_family'].nunique()}/{runs['task_family'].nunique()}")

# Aggregate by task family and model
task_model_performance = runs_with_compute.groupby(['task_family', 'model']).agg({
    'score_cont': 'mean',
    'compute': 'first',
    'compute_source': 'first'
}).reset_index()

print(f"\nTask-model combinations: {len(task_model_performance)}")

# For each task family, fit scaling curve
print(f"\n{'='*100}")
print("TASK-LEVEL SCALING ANALYSIS:")
print(f"{'='*100}")

task_results = []

# Simple power law: score = a * log(compute) + b
def log_scaling(compute, a, b):
    return a * np.log10(compute) + b

for task_family in task_model_performance['task_family'].unique():
    task_data = task_model_performance[task_model_performance['task_family'] == task_family].copy()
    
    if len(task_data) < 3:  # Need at least 3 points
        continue
    
    X = task_data['compute'].values
    Y = task_data['score_cont'].values
    
    # Check if there's variation
    if Y.std() < 0.01:  # Nearly constant
        continue
    
    try:
        # Fit power law
        popt, _ = curve_fit(log_scaling, X, Y, maxfev=5000)
        a_fit, b_fit = popt
        
        # Predict scores
        Y_pred = log_scaling(X, *popt)
        
        # Calculate R²
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # LUCR: Actual vs predicted
        # Positive LUCR = better than scaling predicts (algorithmic progress)
        lucr_values = Y - Y_pred
        mean_lucr = lucr_values.mean()
        
        task_results.append({
            'task_family': task_family,
            'n_models': len(task_data),
            'n_known': (task_data['compute_source'] == 'known').sum(),
            'n_estimated': (task_data['compute_source'] == 'estimated').sum(),
            'slope': a_fit,
            'intercept': b_fit,
            'r2': r2,
            'mean_score': Y.mean(),
            'score_range': Y.max() - Y.min(),
            'mean_lucr': mean_lucr,
            'lucr_std': lucr_values.std()
        })
        
    except Exception as e:
        pass  # Skip tasks where fit fails

task_results_df = pd.DataFrame(task_results)

# Filter to tasks with decent fits
task_results_df = task_results_df[task_results_df['r2'] > 0.3].copy()

print(f"\nTasks with reliable scaling fits (R² > 0.3): {len(task_results_df)}")

# Sort by mean LUCR (algorithmic progress indicator)
task_results_df = task_results_df.sort_values('mean_lucr', ascending=False)

# Categorize tasks
task_results_df['category'] = 'Neutral'
task_results_df.loc[task_results_df['mean_lucr'] > 0.05, 'category'] = 'Algorithmic Progress'
task_results_df.loc[task_results_df['mean_lucr'] < -0.05, 'category'] = 'Scaling Limited'

# Count by category
categories = task_results_df['category'].value_counts()

print(f"\n{'='*100}")
print("TASK CATEGORIES:")
print(f"{'='*100}")
for cat, count in categories.items():
    print(f"  {cat}: {count} tasks")

# Top algorithmic progress tasks
print(f"\n{'='*100}")
print("TOP ALGORITHMIC PROGRESS TASKS:")
print(f"  (Outperforming compute scaling predictions)")
print(f"{'='*100}")

top_algo = task_results_df[task_results_df['category'] == 'Algorithmic Progress'].head(10)
for _, row in top_algo.iterrows():
    print(f"\n{row['task_family']}:")
    print(f"  LUCR: +{row['mean_lucr']:.3f} (models {row['mean_lucr']*100:.1f}% better than scaling predicts)")
    print(f"  Mean score: {row['mean_score']:.2f}")
    print(f"  Models: {row['n_models']} ({row['n_known']} known, {row['n_estimated']} estimated)")
    print(f"  R²: {row['r2']:.3f}")

# Scaling limited tasks
print(f"\n{'='*100}")
print("SCALING LIMITED TASKS:")
print(f"  (Underperforming compute scaling predictions)")
print(f"{'='*100}")

scaling_limited = task_results_df[task_results_df['category'] == 'Scaling Limited'].head(10)
for _, row in scaling_limited.iterrows():
    print(f"\n{row['task_family']}:")
    print(f"  LUCR: {row['mean_lucr']:.3f} (models {abs(row['mean_lucr'])*100:.1f}% worse than scaling predicts)")
    print(f"  Mean score: {row['mean_score']:.2f}")
    print(f"  Models: {row['n_models']} ({row['n_known']} known, {row['n_estimated']} estimated)")
    print(f"  R²: {row['r2']:.3f}")

# Save results
task_results_df.to_csv('outputs/task_level_lucr.csv', index=False)
print(f"\n\nSaved to outputs/task_level_lucr.csv")

# Visualize
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: LUCR distribution
ax1 = fig.add_subplot(gs[0, :])

colors = task_results_df['category'].map({
    'Algorithmic Progress': '#06A77D',
    'Neutral': '#2E86AB', 
    'Scaling Limited': '#E63946'
})

bars = ax1.barh(range(len(task_results_df)), task_results_df['mean_lucr'],
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

ax1.axvline(0, color='black', linestyle='--', linewidth=2)
ax1.set_yticks(range(len(task_results_df)))
ax1.set_yticklabels(task_results_df['task_family'], fontsize=7)
ax1.set_xlabel('Mean LUCR (Actual - Predicted Score)', fontsize=12, fontweight='bold')
ax1.set_title('Task-Level Algorithmic Progress (LUCR)', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#06A77D', label='Algorithmic Progress'),
    Patch(facecolor='#2E86AB', label='Neutral'),
    Patch(facecolor='#E63946', label='Scaling Limited')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Plot 2: LUCR vs Mean Score
ax2 = fig.add_subplot(gs[1, 0])

for category in ['Algorithmic Progress', 'Neutral', 'Scaling Limited']:
    data = task_results_df[task_results_df['category'] == category]
    ax2.scatter(data['mean_score'], data['mean_lucr'],
               label=category, alpha=0.7, s=100, edgecolors='black', linewidth=1)

ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Mean Task Score', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean LUCR', fontsize=12, fontweight='bold')
ax2.set_title('LUCR vs Task Difficulty', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Scaling slope distribution
ax3 = fig.add_subplot(gs[1, 1])

ax3.hist(task_results_df['slope'], bins=15, color='#2E86AB', alpha=0.7, edgecolor='black')
ax3.axvline(task_results_df['slope'].median(), color='red', linestyle='--', linewidth=2,
           label=f'Median: {task_results_df["slope"].median():.3f}')

ax3.set_xlabel('Scaling Slope (Δscore / Δlog10(compute))', fontsize=12, fontweight='bold')
ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
ax3.set_title('Task Scaling Sensitivity', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Category breakdown
ax4 = fig.add_subplot(gs[2, 0])

category_counts = task_results_df['category'].value_counts()
colors_pie = ['#06A77D', '#2E86AB', '#E63946']

wedges, texts, autotexts = ax4.pie(category_counts.values, labels=category_counts.index,
                                    colors=colors_pie, autopct='%1.1f%%',
                                    startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})

ax4.set_title('Task Distribution by Category', fontsize=14, fontweight='bold', pad=15)

# Plot 5: Uncertainty impact
ax5 = fig.add_subplot(gs[2, 1])

known_lucr = task_results_df.groupby('n_known')['mean_lucr'].mean()
estimated_lucr = task_results_df.groupby('n_estimated')['mean_lucr'].mean()

ax5.text(0.1, 0.8, f"Tasks with Known Compute:\n  Mean LUCR: {task_results_df[task_results_df['n_known'] > 0]['mean_lucr'].mean():.3f}",
        transform=ax5.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax5.text(0.1, 0.5, f"Tasks with Estimated Compute:\n  Mean LUCR: {task_results_df[task_results_df['n_estimated'] > 0]['mean_lucr'].mean():.3f}",
        transform=ax5.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

ax5.text(0.1, 0.2, f"Uncertainty Note:\nEstimated compute has ±5x range\nLUCR values may shift but\nrelative ordering likely stable",
        transform=ax5.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

ax5.axis('off')

plt.savefig('outputs/task_level_lucr_analysis.png', dpi=300, bbox_inches='tight')
print("Saved visualization to outputs/task_level_lucr_analysis.png")

print(f"\n{'='*100}")
print("✅ TASK-LEVEL LUCR COMPLETE")
print(f"{'='*100}")

print(f"\n💡 KEY FINDINGS:")
print(f"   {len(task_results_df)} tasks with reliable scaling curves")
print(f"   {categories.get('Algorithmic Progress', 0)} tasks show algorithmic progress (LUCR > 0.05)")
print(f"   {categories.get('Scaling Limited', 0)} tasks hitting scaling limits (LUCR < -0.05)")
print(f"\n   → This shows which capabilities benefit from innovation vs just compute")


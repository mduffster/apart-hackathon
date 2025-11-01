"""
Step 3: Fit Kaplan-style Scaling Law
Goal: Fit METR_score = f(training_compute) using empirical data
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Load METR runs data
print("Loading METR data...")
metr_runs = []
with open('data/METR/all_runs.jsonl', 'r') as f:
    for line in f:
        metr_runs.append(json.loads(line))

metr_df = pd.DataFrame(metr_runs)

# Load model matches (to get compute/params data)
matches = pd.read_csv('outputs/metr_ai_model_matches.csv')

# Create lookup: model alias → compute, params
model_lookup = dict(zip(matches['metr_alias'], 
                       zip(matches['training_compute_flops'], matches['parameters'])))

print(f"Loaded {len(metr_df)} task runs across {len(metr_df['alias'].unique())} models")

# Add compute/params to each run
metr_df['training_compute_flops'] = metr_df['alias'].map(
    lambda x: model_lookup.get(x, (None, None))[0]
)
metr_df['parameters'] = metr_df['alias'].map(
    lambda x: model_lookup.get(x, (None, None))[1]
)

# Filter to runs with compute data
metr_with_compute = metr_df[metr_df['training_compute_flops'].notna()].copy()

print(f"\nRuns with compute data: {len(metr_with_compute)}")
print(f"Models: {metr_with_compute['alias'].nunique()}")
print(f"Task families: {metr_with_compute['task_family'].nunique()}")

# Aggregate scores by model and task family (taking mean if multiple runs)
print("\nAggregating scores by model and task family...")
model_task_scores = metr_with_compute.groupby(['alias', 'task_family', 'training_compute_flops']).agg({
    'score_cont': 'mean',
    'parameters': 'first'
}).reset_index()

print(f"Aggregated to {len(model_task_scores)} model-task combinations")

# Further aggregate across all tasks to get overall model performance
print("\nCalculating overall model performance (frontier-weighted)...")

# Use frontier weighting: (1 - score)^2
model_task_scores['frontier_weight'] = (1 - model_task_scores['score_cont']) ** 2

model_performance = model_task_scores.groupby(['alias', 'training_compute_flops', 'parameters']).apply(
    lambda x: np.average(x['score_cont'], weights=x['frontier_weight'])
).reset_index(name='overall_score')

print(f"\nModel performance summary:")
print(model_performance.sort_values('training_compute_flops'))

# Define scaling law functional forms

def power_law(x, a, b, c):
    """score = a * compute^b + c"""
    return a * np.power(x, b) + c

def log_law(x, a, b):
    """score = a * log(compute) + b"""
    return a * np.log(x) + b

# Fit power law (Kaplan-style)
print("\n" + "="*100)
print("FITTING SCALING LAWS:")
print("="*100)

X = model_performance['training_compute_flops'].values
y = model_performance['overall_score'].values

# Calculate total sum of squares (for R² calculation)
ss_tot = np.sum((y - y.mean()) ** 2)

print(f"\nData points: {len(X)}")
print(f"  Compute range: {X.min():.2e} to {X.max():.2e} FLOPs")
print(f"  Score range: {y.min():.4f} to {y.max():.4f}")

# Fit power law
try:
    # Use looser initial guess and bounds
    p0_power = [1e-9, 0.1, 0.01]  # Start with very small coefficient
    bounds_power = ([0, 0, -1], [1, 1, 2])  # Allow wider range
    
    popt_power, pcov_power = curve_fit(power_law, X, y, p0=p0_power, bounds=bounds_power, maxfev=50000)
    a_power, b_power, c_power = popt_power
    
    # Calculate R²
    y_pred_power = power_law(X, *popt_power)
    ss_res_power = np.sum((y - y_pred_power) ** 2)
    r2_power = 1 - (ss_res_power / ss_tot)
    
    print(f"\n✓ Power Law: score = {a_power:.6e} * compute^{b_power:.6f} + {c_power:.6f}")
    print(f"  R² = {r2_power:.4f}")
    
except Exception as e:
    print(f"\n✗ Power law fit failed: {e}")
    popt_power = None
    r2_power = None

# Fit logarithmic law
try:
    # Use log10 for better numerical stability
    X_log10 = np.log10(X)
    
    # Simple linear fit: score = a * log10(compute) + b
    coeffs = np.polyfit(X_log10, y, 1)
    a_log, b_log = coeffs
    popt_log = coeffs
    
    # Calculate R²
    y_pred_log = a_log * X_log10 + b_log
    ss_res_log = np.sum((y - y_pred_log) ** 2)
    r2_log = 1 - (ss_res_log / ss_tot)
    
    print(f"\n✓ Log Law: score = {a_log:.6f} * log10(compute) + {b_log:.6f}")
    print(f"  R² = {r2_log:.4f}")
    
    # Define log law function for predictions
    def log_law_predict(x):
        return a_log * np.log10(x) + b_log
    
except Exception as e:
    print(f"\n✗ Log law fit failed: {e}")
    popt_log = None
    r2_log = None

# Choose best fit
if popt_power is not None and (popt_log is None or r2_power > r2_log):
    best_fit = 'power'
    best_params = popt_power
    print(f"\n🏆 Best fit: Power Law (R² = {r2_power:.4f})")
elif popt_log is not None:
    best_fit = 'log'
    best_params = popt_log
    print(f"\n🏆 Best fit: Log Law (R² = {r2_log:.4f})")
else:
    print("\n⚠️ No successful fits!")
    best_fit = None

# Save scaling law parameters
scaling_law_params = {
    'power_law': {
        'a': popt_power[0] if popt_power is not None else None,
        'b': popt_power[1] if popt_power is not None else None,
        'c': popt_power[2] if popt_power is not None else None,
        'r2': r2_power
    },
    'log_law': {
        'a': popt_log[0] if popt_log is not None else None,
        'b': popt_log[1] if popt_log is not None else None,
        'r2': r2_log
    },
    'best_fit': best_fit
}

import json
with open('outputs/scaling_law_params.json', 'w') as f:
    json.dump(scaling_law_params, f, indent=2)

print(f"\nSaved scaling law parameters to outputs/scaling_law_params.json")

# Plot: Compute vs Performance with scaling law fit
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot of actual data
colors = ['#2E86AB', '#A23B72', '#F18F01']
for i, (_, row) in enumerate(model_performance.iterrows()):
    ax.scatter(row['training_compute_flops'], row['overall_score'], 
              s=200, alpha=0.7, color=colors[i % len(colors)], edgecolors='black', linewidth=1.5,
              zorder=10)
    ax.annotate(row['alias'], 
               xy=(row['training_compute_flops'], row['overall_score']),
               xytext=(10, 0), textcoords='offset points',
               fontsize=9, fontweight='bold')

# Plot scaling law fits
X_smooth = np.logspace(np.log10(X.min()), np.log10(X.max() * 100), 500)

if popt_power is not None:
    y_power_smooth = power_law(X_smooth, *popt_power)
    ax.plot(X_smooth, y_power_smooth, '--', linewidth=2.5, color='#E63946', 
           label=f'Power Law (R²={r2_power:.3f})', alpha=0.8)

if popt_log is not None:
    y_log_smooth = log_law_predict(X_smooth)
    ax.plot(X_smooth, y_log_smooth, ':', linewidth=2.5, color='#06A77D',
           label=f'Log Law (R²={r2_log:.3f})', alpha=0.8)

ax.set_xscale('log')
ax.set_xlabel('Training Compute (FLOPs)', fontsize=13, fontweight='bold')
ax.set_ylabel('Overall METR Score (Frontier-Weighted)', fontsize=13, fontweight='bold')
ax.set_title('Kaplan-Style Scaling Law: METR Performance vs Training Compute', 
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim([0, max(y.max() * 1.1, 1)])

plt.tight_layout()
plt.savefig('outputs/kaplan_scaling_law.png', dpi=300, bbox_inches='tight')
print(f"Saved scaling law plot to outputs/kaplan_scaling_law.png")

# Display predictions for hypothetical future compute
print("\n" + "="*100)
print("PREDICTIONS FOR FUTURE COMPUTE:")
print("="*100)

future_compute_values = [1e26, 1e27, 1e28]  # 10x, 100x, 1000x current frontier

if best_fit == 'power':
    for compute in future_compute_values:
        predicted_score = power_law(compute, *popt_power)
        print(f"  {compute:.1e} FLOPs → Predicted score: {predicted_score:.3f}")
elif best_fit == 'log':
    for compute in future_compute_values:
        predicted_score = log_law_predict(compute)
        print(f"  {compute:.1e} FLOPs → Predicted score: {predicted_score:.3f}")

print("\n✅ Scaling law fitted successfully!")


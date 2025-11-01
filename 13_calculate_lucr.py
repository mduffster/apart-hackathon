"""
Calculate LUCR: Loss to Utility Conversion Rate
Compare Kaplan "horsepower" predictions to actual METR frontier performance
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

# Load Kaplan horsepower parameters
with open('outputs/kaplan_horsepower_params.json', 'r') as f:
    horsepower_params = json.load(f)

# Reconstruct horsepower functions
L_inf = horsepower_params['kaplan_constants']['L_inf']
A = horsepower_params['kaplan_constants']['A']
alpha = horsepower_params['kaplan_constants']['alpha']

k_fit = horsepower_params['sigmoid_fit']['k']
L_0_fit = horsepower_params['sigmoid_fit']['L_0']
ECI_min_fit = horsepower_params['sigmoid_fit']['ECI_min']
ECI_max_fit = horsepower_params['sigmoid_fit']['ECI_max']

eci_to_metr_slope = horsepower_params['eci_to_metr']['slope']
eci_to_metr_intercept = horsepower_params['eci_to_metr']['intercept']

def kaplan_loss(compute):
    return L_inf + A * np.power(compute, -alpha)

def sigmoid_loss_to_eci(loss):
    sigmoid = 1.0 / (1.0 + np.exp(-k_fit * (L_0_fit - loss)))
    return ECI_min_fit + (ECI_max_fit - ECI_min_fit) * sigmoid

def kaplan_horsepower_metr(compute):
    loss = kaplan_loss(compute)
    eci = sigmoid_loss_to_eci(loss)
    metr = eci_to_metr_slope * eci + eci_to_metr_intercept
    return metr

# Load METR data and build frontier time series
print("Loading METR data...")
metr_runs = []
with open('data/METR/all_runs.jsonl', 'r') as f:
    for line in f:
        metr_runs.append(json.loads(line))

metr_df = pd.DataFrame(metr_runs)

# Load model matches to get release dates
matches = pd.read_csv('outputs/metr_ai_model_matches.csv')
matches['release_date'] = pd.to_datetime(matches['release_date'])

# Add release dates to METR
alias_to_date = dict(zip(matches['metr_alias'], matches['release_date']))
alias_to_compute = dict(zip(matches['metr_alias'], matches['training_compute_flops']))

metr_df['release_date'] = metr_df['alias'].map(alias_to_date)
metr_df['training_compute'] = metr_df['alias'].map(alias_to_compute)

# Filter to runs with dates
metr_with_dates = metr_df[metr_df['release_date'].notna()].copy()

print(f"METR runs with dates: {len(metr_with_dates)}")

# Build monthly time series of METR frontier
min_date = metr_with_dates['release_date'].min()
max_date = pd.Timestamp.now()
date_range = pd.date_range(start=min_date, end=max_date, freq='MS')

frontier_series = []

for date in date_range:
    # Get runs from models released by this date
    available_runs = metr_with_dates[metr_with_dates['release_date'] <= date]
    
    if len(available_runs) == 0:
        continue
    
    # Calculate frontier-weighted score
    best_scores_per_task = available_runs.groupby('task_family')['score_cont'].max()
    weights = (1 - best_scores_per_task) ** 2
    frontier_weighted = np.average(best_scores_per_task, weights=weights)
    
    # Get max compute at this time
    max_compute = available_runs['training_compute'].max()
    
    # Calculate Kaplan prediction
    kaplan_pred = kaplan_horsepower_metr(max_compute) if pd.notna(max_compute) else np.nan
    
    # Calculate LUCR
    lucr = kaplan_pred - frontier_weighted if pd.notna(kaplan_pred) else np.nan
    
    frontier_series.append({
        'date': date,
        'metr_actual': frontier_weighted,
        'max_compute': max_compute,
        'kaplan_predicted': kaplan_pred,
        'lucr': lucr
    })

frontier_df = pd.DataFrame(frontier_series)

# Remove rows with NaN LUCR
frontier_df = frontier_df[frontier_df['lucr'].notna()]

print(f"\n✓ Built frontier time series: {len(frontier_df)} time points")

# Display key stats
print("\n" + "="*100)
print("LUCR ANALYSIS:")
print("="*100)

print(f"\nCurrent state ({frontier_df['date'].iloc[-1].date()}):")
print(f"  Max compute: {frontier_df['max_compute'].iloc[-1]:.2e} FLOPs")
print(f"  METR actual: {frontier_df['metr_actual'].iloc[-1]:.4f} ({frontier_df['metr_actual'].iloc[-1]*100:.2f}%)")
print(f"  Kaplan predicted: {frontier_df['kaplan_predicted'].iloc[-1]:.4f} ({frontier_df['kaplan_predicted'].iloc[-1]*100:.2f}%)")
print(f"  LUCR: {frontier_df['lucr'].iloc[-1]:.4f}")

if frontier_df['lucr'].iloc[-1] > 0:
    print(f"  → INEFFICIENT: Kaplan predicts {abs(frontier_df['lucr'].iloc[-1])*100:.2f}% more capability than we achieve")
else:
    print(f"  → EFFICIENT: We achieve {abs(frontier_df['lucr'].iloc[-1])*100:.2f}% more capability than Kaplan predicts")

# Analyze LUCR trend
lucr_trend_coeffs = np.polyfit(range(len(frontier_df)), frontier_df['lucr'].values, 1)
lucr_slope = lucr_trend_coeffs[0]

print(f"\nLUCR trend: {lucr_slope:.6f} per month")
if lucr_slope > 0:
    print(f"  → LUCR is INCREASING: Efficiency gap is growing")
    print(f"  → Raw compute scaling is not translating to capability gains")
else:
    print(f"  → LUCR is DECREASING: Efficiency gap is shrinking")
    print(f"  → Algorithmic improvements are helping")

# Save results
frontier_df.to_csv('outputs/lucr_time_series.csv', index=False)
print(f"\nSaved LUCR time series to outputs/lucr_time_series.csv")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# Plot 1: Actual vs Predicted Capability
ax1 = axes[0]
ax1.plot(frontier_df['date'], frontier_df['kaplan_predicted']*100, 
         'o-', linewidth=2.5, markersize=6, color='#E63946', label='Kaplan "Horsepower"', alpha=0.8)
ax1.plot(frontier_df['date'], frontier_df['metr_actual']*100, 
         's-', linewidth=2.5, markersize=6, color='#06A77D', label='Actual METR Frontier', alpha=0.8)
ax1.fill_between(frontier_df['date'], 
                 frontier_df['kaplan_predicted']*100, 
                 frontier_df['metr_actual']*100,
                 alpha=0.2, color='gray', label='LUCR Gap')

ax1.axhline(100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Human-level (100%)')
ax1.axhline(90, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Near-AGI (90%)')

ax1.set_ylabel('METR Frontier Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('Kaplan "Horsepower" vs Actual Capability', fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, max(10, frontier_df['metr_actual'].max()*120)])

# Plot 2: LUCR over time
ax2 = axes[1]
colors = ['#E63946' if x > 0 else '#06A77D' for x in frontier_df['lucr']]
ax2.bar(frontier_df['date'], frontier_df['lucr']*100, width=20, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax2.axhline(0, color='black', linewidth=2, linestyle='--')

# Add trend line
trend_line = lucr_trend_coeffs[0] * np.arange(len(frontier_df)) + lucr_trend_coeffs[1]
ax2.plot(frontier_df['date'], trend_line*100, 'r--', linewidth=2, alpha=0.6, 
        label=f'Trend ({lucr_slope*100:.4f}%/month)')

ax2.set_ylabel('LUCR (%)', fontsize=12, fontweight='bold')
ax2.set_title('Loss to Utility Conversion Rate (LUCR) Over Time', fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Compute frontier
ax3 = axes[2]
ax3.plot(frontier_df['date'], frontier_df['max_compute'], 
         'o-', linewidth=2.5, markersize=6, color='#2E86AB')
ax3.set_yscale('log')
ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frontier Compute (FLOPs)', fontsize=12, fontweight='bold')
ax3.set_title('Compute Frontier Evolution', fontsize=14, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('outputs/lucr_analysis.png', dpi=300, bbox_inches='tight')
print("Saved LUCR analysis plot to outputs/lucr_analysis.png")

print("\n" + "="*100)
print("✅ LUCR CALCULATION COMPLETE!")
print("="*100)
print("\nNext step: Use LUCR trend to project compute needed for METR = 0.9 (near-AGI)")


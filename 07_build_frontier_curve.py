"""
Step 2: Build Frontier Time Series
Goal: Identify frontier models at each time point (within 1 OOM of max compute/params)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load matched models
matches = pd.read_csv('outputs/metr_ai_model_matches.csv')

# Convert date to datetime
matches['release_date'] = pd.to_datetime(matches['release_date'])

# Sort by date
matches = matches.sort_values('release_date')

print("Matched models (chronological):")
print("="*100)
for _, row in matches.iterrows():
    print(f"{row['release_date'].strftime('%Y-%m-%d')} | {row['metr_alias']:30s} | "
          f"{row['training_compute_flops']:.2e} FLOPs | {row['parameters']:.2e} params")

# Create time series with monthly granularity
min_date = matches['release_date'].min()
max_date = pd.Timestamp.now()

# Generate monthly date range
date_range = pd.date_range(start=min_date, end=max_date, freq='MS')  # Month start

print(f"\n\nBuilding frontier time series from {min_date.date()} to {max_date.date()}")
print(f"Time points: {len(date_range)} months")

# For each time point, identify frontier models
# Frontier = models within 1 order of magnitude (10x) of max compute/params at that time

frontier_data = []

for date in date_range:
    # Get all models released by this date
    available_models = matches[matches['release_date'] <= date]
    
    if len(available_models) == 0:
        continue
    
    # Find maximum compute and parameters at this time
    max_compute = available_models['training_compute_flops'].max()
    max_params = available_models['parameters'].max()
    
    # Define frontier: within 1 OOM (10x) of max
    compute_threshold = max_compute / 10
    params_threshold = max_params / 10
    
    # Models that are frontier on EITHER compute OR params
    is_frontier_compute = available_models['training_compute_flops'] >= compute_threshold
    is_frontier_params = available_models['parameters'] >= params_threshold
    is_frontier = is_frontier_compute | is_frontier_params
    
    frontier_models = available_models[is_frontier]
    
    # Store frontier stats for this time point
    frontier_data.append({
        'date': date,
        'max_compute_flops': max_compute,
        'max_parameters': max_params,
        'compute_threshold': compute_threshold,
        'params_threshold': params_threshold,
        'num_frontier_models': len(frontier_models),
        'frontier_model_names': ', '.join(frontier_models['metr_alias'].unique())
    })

frontier_df = pd.DataFrame(frontier_data)

# Save frontier time series
frontier_df.to_csv('outputs/frontier_time_series.csv', index=False)
print(f"\nSaved frontier time series to outputs/frontier_time_series.csv")

# Display key frontier transitions
print("\n" + "="*100)
print("KEY FRONTIER TRANSITIONS:")
print("="*100)

# Find when max compute increases
frontier_df['compute_increased'] = frontier_df['max_compute_flops'].diff() > 0

transitions = frontier_df[frontier_df['compute_increased']]

for _, row in transitions.iterrows():
    print(f"\n{row['date'].strftime('%Y-%m-%d')}:")
    print(f"  Max compute: {row['max_compute_flops']:.2e} FLOPs")
    print(f"  Max params: {row['max_parameters']:.2e}")
    print(f"  Frontier models ({row['num_frontier_models']}): {row['frontier_model_names']}")

# Plot frontier evolution
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Compute frontier over time
ax1 = axes[0]
ax1.plot(frontier_df['date'], frontier_df['max_compute_flops'], 
         'o-', color='#2E86AB', linewidth=2, markersize=8, label='Max Compute')
ax1.plot(frontier_df['date'], frontier_df['compute_threshold'], 
         '--', color='#A23B72', linewidth=1.5, alpha=0.7, label='Frontier Threshold (÷10)')
ax1.set_yscale('log')
ax1.set_ylabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax1.set_title('Compute Frontier Evolution Over Time', fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Annotate model releases
for _, row in matches.iterrows():
    if row['training_compute_flops'] >= row['training_compute_flops'] * 0.5:  # Only annotate high-compute models
        ax1.annotate(row['metr_alias'], 
                    xy=(row['release_date'], row['training_compute_flops']),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=8, ha='center', rotation=0,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# Plot 2: Parameters frontier over time
ax2 = axes[1]
ax2.plot(frontier_df['date'], frontier_df['max_parameters'], 
         'o-', color='#F18F01', linewidth=2, markersize=8, label='Max Parameters')
ax2.plot(frontier_df['date'], frontier_df['params_threshold'], 
         '--', color='#C73E1D', linewidth=1.5, alpha=0.7, label='Frontier Threshold (÷10)')
ax2.set_yscale('log')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Parameters', fontsize=12, fontweight='bold')
ax2.set_title('Parameter Frontier Evolution Over Time', fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/frontier_evolution.png', dpi=300, bbox_inches='tight')
print(f"\nSaved frontier evolution plot to outputs/frontier_evolution.png")

# Summary statistics
print("\n" + "="*100)
print("FRONTIER SUMMARY:")
print("="*100)
print(f"\nCompute growth:")
initial_compute = frontier_df['max_compute_flops'].iloc[0]
final_compute = frontier_df['max_compute_flops'].iloc[-1]
time_span_years = (frontier_df['date'].iloc[-1] - frontier_df['date'].iloc[0]).days / 365.25
print(f"  Initial (2019): {initial_compute:.2e} FLOPs")
print(f"  Current (2024): {final_compute:.2e} FLOPs")
print(f"  Total increase: {final_compute/initial_compute:.1f}x")
print(f"  Doubling time: {time_span_years * np.log(2) / np.log(final_compute/initial_compute):.2f} months")

print(f"\nParameter growth:")
initial_params = frontier_df['max_parameters'].iloc[0]
final_params = frontier_df['max_parameters'].iloc[-1]
print(f"  Initial (2019): {initial_params:.2e}")
print(f"  Current (2024): {final_params:.2e}")
print(f"  Total increase: {final_params/initial_params:.1f}x")
print(f"  Doubling time: {time_span_years * np.log(2) / np.log(final_params/initial_params):.2f} months")


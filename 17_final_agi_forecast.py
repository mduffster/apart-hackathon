"""
Final AGI Forecast: Project to METR = 0.9
Combining:
- LUCR efficiency improvements (-0.16 ECI per model)
- Compute scaling
- ECI → METR mapping
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from datetime import datetime, timedelta

# Load data
print("Loading data...")
lucr_eci = pd.read_csv('outputs/lucr_eci_scale.csv')
lucr_eci['Release date'] = pd.to_datetime(lucr_eci['Release date'])

# Load ECI→METR mapping
with open('outputs/eci_to_metr_mapping.json', 'r') as f:
    mapping = json.load(f)

eci_to_metr_slope = mapping['slope']
eci_to_metr_intercept = mapping['intercept']

def eci_to_metr(eci):
    return eci_to_metr_slope * eci + eci_to_metr_intercept

# Load Kaplan horsepower function
with open('outputs/kaplan_horsepower_params.json', 'r') as f:
    horsepower_params = json.load(f)

L_inf = horsepower_params['kaplan_constants']['L_inf']
A = horsepower_params['kaplan_constants']['A']
alpha = horsepower_params['kaplan_constants']['alpha']
k_fit = horsepower_params['sigmoid_fit']['k']
L_0_fit = horsepower_params['sigmoid_fit']['L_0']
ECI_min_fit = horsepower_params['sigmoid_fit']['ECI_min']
ECI_max_fit = horsepower_params['sigmoid_fit']['ECI_max']

def kaplan_loss(compute):
    return L_inf + A * np.power(compute, -alpha)

def sigmoid_loss_to_eci(loss):
    sigmoid = 1.0 / (1.0 + np.exp(-k_fit * (L_0_fit - loss)))
    return ECI_min_fit + (ECI_max_fit - ECI_min_fit) * sigmoid

def kaplan_horsepower_eci(compute):
    loss = kaplan_loss(compute)
    return sigmoid_loss_to_eci(loss)

# Analyze trends
print("\nAnalyzing trends...")

# LUCR efficiency trend (how much we beat Kaplan per model)
time_idx = range(len(lucr_eci))
lucr_diff_trend = np.polyfit(time_idx, lucr_eci['lucr_eci_diff'].values, 1)
lucr_efficiency_improvement = -lucr_diff_trend[0]  # Negative because we're BELOW Kaplan (more efficient)

print(f"LUCR efficiency improvement: {lucr_efficiency_improvement:.4f} ECI per model")

# Current state
current_compute = lucr_eci['Training compute (FLOP)'].iloc[-1]
current_eci_actual = lucr_eci['ECI Score'].iloc[-1]
current_date = lucr_eci['Release date'].max()

print(f"\nCurrent state ({current_date.date()}):")
print(f"  Compute: {current_compute:.2e} FLOPs")
print(f"  ECI: {current_eci_actual:.2f}")
print(f"  METR: {eci_to_metr(current_eci_actual):.4f} ({eci_to_metr(current_eci_actual)*100:.2f}%)")

# Project forward
print("\n" + "="*100)
print("PROJECTING TO AGI (METR = 0.9):")
print("="*100)

# Assumptions
compute_doubling_months = 6  # Compute doubles every 6 months
models_per_year = 12  # ~1 frontier model per month

# Calculate how much ECI improvement we need
target_metr = 0.9
target_eci = (target_metr - eci_to_metr_intercept) / eci_to_metr_slope

print(f"\nTarget: METR = {target_metr:.2f} → ECI = {target_eci:.1f}")
print(f"Current: ECI = {current_eci_actual:.1f}")
print(f"Gap: {target_eci - current_eci_actual:.1f} ECI points")

# Scenario: Project with efficiency improvements
projections = []

for years_ahead in [0.5, 1, 2, 3, 5, 10, 15, 20]:
    months = years_ahead * 12
    
    # Compute growth
    compute_multiplier = 2 ** (months / compute_doubling_months)
    future_compute = current_compute * compute_multiplier
    
    # Kaplan baseline (no efficiency improvement)
    kaplan_eci = kaplan_horsepower_eci(future_compute)
    
    # With efficiency improvements
    num_models = years_ahead * models_per_year
    efficiency_boost = lucr_efficiency_improvement * num_models
    
    # Actual ECI = Kaplan + efficiency improvements
    projected_eci = kaplan_eci + efficiency_boost
    projected_metr = eci_to_metr(projected_eci)
    
    projections.append({
        'years': years_ahead,
        'date': current_date + timedelta(days=365*years_ahead),
        'compute': future_compute,
        'compute_multiplier': compute_multiplier,
        'kaplan_eci': kaplan_eci,
        'efficiency_boost': efficiency_boost,
        'projected_eci': projected_eci,
        'projected_metr': projected_metr
    })
    
    print(f"\n{years_ahead:.1f} years from now ({(current_date + timedelta(days=365*years_ahead)).date()}):")
    print(f"  Compute: {future_compute:.2e} FLOPs ({compute_multiplier:.0f}x)")
    print(f"  Kaplan ECI: {kaplan_eci:.1f}")
    print(f"  Efficiency boost: +{efficiency_boost:.1f} ECI")
    print(f"  Projected ECI: {projected_eci:.1f}")
    print(f"  Projected METR: {projected_metr:.4f} ({projected_metr*100:.1f}%)")
    
    if 0.85 <= projected_metr <= 0.95:
        print(f"  ✓✓✓ NEAR-AGI ACHIEVED!")

projections_df = pd.DataFrame(projections)

# Find when we hit METR = 0.9
target_idx = np.argmin(np.abs(projections_df['projected_metr'] - target_metr))
target_projection = projections_df.iloc[target_idx]

print("\n" + "="*100)
print("AGI TIMELINE:")
print("="*100)
print(f"\nMETR = {target_metr:.1f} achieved in ~{target_projection['years']:.1f} years")
print(f"  Date: {target_projection['date'].date()}")
print(f"  Compute needed: {target_projection['compute']:.2e} FLOPs ({target_projection['compute_multiplier']:.0f}x current)")
print(f"  ECI: {target_projection['projected_eci']:.1f}")
print(f"  Breakdown:")
print(f"    - Kaplan contribution: {target_projection['kaplan_eci']:.1f} ECI")
print(f"    - Efficiency improvements: +{target_projection['efficiency_boost']:.1f} ECI")

# Save
projections_df.to_csv('outputs/agi_timeline_projection.csv', index=False)
print(f"\nSaved projections to outputs/agi_timeline_projection.csv")

# Visualize
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: METR projection over time
ax1 = fig.add_subplot(gs[0, :])

# Historical METR (from our frontier analysis)
metr_frontier = pd.read_csv('outputs/lucr_time_series.csv')
metr_frontier['date'] = pd.to_datetime(metr_frontier['date'])

ax1.plot(metr_frontier['date'], metr_frontier['metr_actual']*100,
        'o-', linewidth=2.5, markersize=8, color='#06A77D', label='Historical METR Frontier', zorder=10)

# Projections
ax1.plot(projections_df['date'], projections_df['projected_metr']*100,
        's-', linewidth=3, markersize=10, color='#E63946', label='Projected METR (with efficiency gains)', zorder=10)

# Target line
ax1.axhline(90, color='orange', linestyle='--', linewidth=2.5, alpha=0.7, label='Near-AGI (90%)')
ax1.axhline(100, color='green', linestyle='--', linewidth=2.5, alpha=0.7, label='Human-level (100%)')

# Mark AGI date
ax1.scatter([target_projection['date']], [target_projection['projected_metr']*100],
           s=500, marker='*', color='gold', edgecolors='black', linewidth=3,
           label=f'AGI: {target_projection["date"].date()}', zorder=15)

ax1.set_ylabel('METR Frontier Score (%)', fontsize=13, fontweight='bold')
ax1.set_title('AGI Forecast: Path to METR = 90%', fontsize=16, fontweight='bold', pad=20)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 105])

# Plot 2: ECI breakdown
ax2 = fig.add_subplot(gs[1, 0])

bar_width = 0.35
x_pos = np.arange(len(projections_df))

ax2.bar(x_pos - bar_width/2, projections_df['kaplan_eci'], bar_width,
       label='Kaplan Baseline', color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.bar(x_pos + bar_width/2, projections_df['projected_eci'], bar_width,
       label='With Efficiency Gains', color='#F18F01', alpha=0.7, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax2.set_ylabel('ECI Score', fontsize=12, fontweight='bold')
ax2.set_title('Kaplan vs Projected ECI', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{y:.1f}' for y in projections_df['years']])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Efficiency contribution
ax3 = fig.add_subplot(gs[1, 1])

ax3.bar(projections_df['years'], projections_df['efficiency_boost'],
       color='#06A77D', alpha=0.7, edgecolor='black', linewidth=1.5)

ax3.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax3.set_ylabel('Efficiency Boost (ECI points)', fontsize=12, fontweight='bold')
ax3.set_title('Algorithmic Efficiency Gains', fontsize=14, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Compute requirements
ax4 = fig.add_subplot(gs[2, 0])

ax4.plot(projections_df['years'], projections_df['compute'],
        'o-', linewidth=3, markersize=10, color='#E63946')

ax4.set_yscale('log')
ax4.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax4.set_ylabel('Compute (FLOPs)', fontsize=12, fontweight='bold')
ax4.set_title('Compute Requirements for AGI', fontsize=14, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, which='both')

# Annotate AGI point
ax4.scatter([target_projection['years']], [target_projection['compute']],
           s=300, marker='*', color='gold', edgecolors='black', linewidth=3, zorder=10)
ax4.text(target_projection['years'], target_projection['compute']*1.5,
        f"AGI: {target_projection['compute']:.1e} FLOPs",
        fontsize=10, fontweight='bold', ha='center')

# Plot 5: METR milestones timeline
ax5 = fig.add_subplot(gs[2, 1])

milestones = [0.5, 0.7, 0.9, 1.0]
milestone_years = []
milestone_dates = []

for milestone in milestones:
    idx = np.argmin(np.abs(projections_df['projected_metr'] - milestone))
    years = projections_df.iloc[idx]['years']
    date = projections_df.iloc[idx]['date']
    milestone_years.append(years)
    milestone_dates.append(date.strftime('%Y-%m'))

colors_milestone = ['#2E86AB', '#06A77D', '#F18F01', '#E63946']
bars = ax5.barh(range(len(milestones)), milestone_years, color=colors_milestone, 
               alpha=0.7, edgecolor='black', linewidth=2)

ax5.set_yticks(range(len(milestones)))
ax5.set_yticklabels([f"METR {m*100:.0f}%" for m in milestones])
ax5.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax5.set_title('Timeline to Capability Milestones', fontsize=14, fontweight='bold', pad=15)
ax5.grid(True, alpha=0.3, axis='x')

# Annotate bars
for i, (years, date) in enumerate(zip(milestone_years, milestone_dates)):
    ax5.text(years + 0.3, i, f'{years:.1f}y\n({date})', 
            va='center', fontsize=9, fontweight='bold')

plt.savefig('outputs/agi_final_forecast.png', dpi=300, bbox_inches='tight')
print("Saved final forecast to outputs/agi_final_forecast.png")

print("\n" + "="*100)
print("✅ AGI FORECAST COMPLETE!")
print("="*100)

print(f"\n🎯 KEY FINDING:")
print(f"   With current efficiency improvement trends, AGI (METR ≥ 0.9) achievable by:")
print(f"   📅 {target_projection['date'].date()}")
print(f"   🔬 ~{target_projection['years']:.1f} years from now")
print(f"   💻 Requiring ~{target_projection['compute_multiplier']:.0f}x current compute")
print(f"\n   This assumes:")
print(f"   • Compute continues doubling every {compute_doubling_months} months")
print(f"   • Efficiency improvements continue at current rate ({lucr_efficiency_improvement:.2f} ECI/model)")
print(f"   • No major breakthroughs or setbacks")


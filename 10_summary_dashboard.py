"""
Create a comprehensive dashboard summarizing the entire AGI forecast framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# Load all data
lucr_df = pd.read_csv('outputs/lucr_time_series.csv')
lucr_df['date'] = pd.to_datetime(lucr_df['date'])

# Filter to valid data points
valid_data = lucr_df[lucr_df['actual_capability'].notna()]

# Create dashboard
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('AGI Forecast Framework: LUCR Analysis Dashboard', 
             fontsize=20, fontweight='bold', y=0.98)

# ==================== Plot 1: Frontier Compute Evolution ====================
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(valid_data['date'], valid_data['max_compute_flops'], 
         'o-', linewidth=3, markersize=10, color='#2E86AB', markeredgecolor='black', markeredgewidth=1.5)
ax1.set_yscale('log')
ax1.set_ylabel('Training Compute (FLOPs)', fontsize=11, fontweight='bold')
ax1.set_title('Frontier Compute Growth', fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, which='both')

# Add growth annotation
initial_compute = valid_data['max_compute_flops'].iloc[0]
final_compute = valid_data['max_compute_flops'].iloc[-1]
growth_factor = final_compute / initial_compute
ax1.text(0.05, 0.95, f'{growth_factor:.0f}x growth\n(2019-2024)', 
        transform=ax1.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# ==================== Plot 2: Capability vs Horsepower ====================
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(valid_data['date'], valid_data['horsepower_predicted'] * 100, 
         'o-', linewidth=2.5, markersize=8, color='#E63946', label='Horsepower (Kaplan)', alpha=0.8)
ax2.plot(valid_data['date'], valid_data['actual_capability'] * 100, 
         's-', linewidth=2.5, markersize=8, color='#06A77D', label='Actual Capability', alpha=0.8)
ax2.fill_between(valid_data['date'], 
                 valid_data['horsepower_predicted'] * 100, 
                 valid_data['actual_capability'] * 100,
                 alpha=0.2, color='gray')
ax2.set_ylabel('METR Score (%)', fontsize=11, fontweight='bold')
ax2.set_title('Predicted vs Actual Capability', fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3)

# ==================== Plot 3: LUCR Over Time ====================
ax3 = fig.add_subplot(gs[1, :])
colors = ['#E63946' if x > 0 else '#06A77D' for x in valid_data['LUCR']]
ax3.bar(valid_data['date'], valid_data['LUCR'] * 100, width=25, color=colors, alpha=0.7, 
       edgecolor='black', linewidth=1.5)
ax3.axhline(0, color='black', linewidth=2.5, linestyle='--', alpha=0.8)
ax3.set_ylabel('LUCR (%)', fontsize=11, fontweight='bold')
ax3.set_title('Loss to Utility Conversion Rate (LUCR): Efficiency Gap Over Time', 
             fontsize=13, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, axis='y')

# Add trend line
z = np.polyfit(range(len(valid_data)), valid_data['LUCR'].values * 100, 1)
p = np.poly1d(z)
trend_line = p(range(len(valid_data)))
ax3.plot(valid_data['date'], trend_line, 'r--', linewidth=2.5, alpha=0.6, label=f'Trend (slope={z[0]:.3f}%/month)')
ax3.legend(fontsize=9, loc='upper left')

# Add annotations
ax3.text(0.98, 0.95, 'INEFFICIENT\n(Underperforming)', 
        transform=ax3.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='#E63946', alpha=0.3))
ax3.text(0.98, 0.05, 'EFFICIENT\n(Overperforming)', 
        transform=ax3.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='#06A77D', alpha=0.3))

# ==================== Plot 4: Summary Statistics ====================
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')

current_lucr = valid_data['LUCR'].iloc[-1] * 100
current_actual = valid_data['actual_capability'].iloc[-1] * 100
current_predicted = valid_data['horsepower_predicted'].iloc[-1] * 100
current_compute = valid_data['max_compute_flops'].iloc[-1]

summary_text = f"""
CURRENT STATE (Oct 2025):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Frontier Compute:     {current_compute:.2e} FLOPs
Horsepower:           {current_predicted:.2f}%
Actual Capability:    {current_actual:.2f}%
LUCR:                 {current_lucr:+.2f}%

EFFICIENCY GAP:       {abs(current_lucr):.1f}% {'BELOW' if current_lucr > 0 else 'ABOVE'} prediction

INTERPRETATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{'⚠️  INEFFICIENT: Compute scaling is NOT' if current_lucr > 0 else '✅ EFFICIENT: Algorithmic improvements'}
{'   translating fully to capability gains.' if current_lucr > 0 else '   are outpacing raw compute.'}

TREND:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LUCR trend: {z[0]:.4f}%/month
{'📈 Efficiency DECLINING over time' if z[0] > 0 else '📉 Efficiency IMPROVING over time'}
{'   → Challenges "doubling every 7 months"' if z[0] > 0 else '   → Supports rapid progress narrative'}
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# ==================== Plot 5: Model Timeline ====================
ax5 = fig.add_subplot(gs[2, 1])

# Load model matches to show timeline
matches = pd.read_csv('outputs/metr_ai_model_matches.csv')
matches['release_date'] = pd.to_datetime(matches['release_date'])
matches = matches.sort_values('release_date')

# Create timeline
model_families = []
model_dates = []
model_names = []

for _, row in matches.iterrows():
    if row['metr_alias'] not in model_names:  # Avoid duplicates
        model_families.append(row['metr_alias'])
        model_dates.append(row['release_date'])
        model_names.append(row['metr_alias'])

y_positions = range(len(model_families))

# Plot timeline
for i, (name, date) in enumerate(zip(model_families, model_dates)):
    ax5.scatter(date, i, s=300, color='#F18F01', edgecolors='black', linewidth=2, zorder=10, marker='o')
    ax5.text(date, i, f'  {name}', fontsize=9, verticalalignment='center', fontweight='bold')
    # Draw horizontal line to show timeline
    if i > 0:
        ax5.plot([model_dates[i-1], date], [i-1, i], 'k-', linewidth=1, alpha=0.3)

ax5.set_yticks([])
ax5.set_xlabel('Release Date', fontsize=11, fontweight='bold')
ax5.set_title('Model Release Timeline', fontsize=13, fontweight='bold', pad=15)
ax5.grid(True, alpha=0.3, axis='x')
ax5.set_ylim(-0.5, len(model_families) - 0.5)

# Save dashboard
plt.savefig('outputs/agi_forecast_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ Saved AGI forecast dashboard to outputs/agi_forecast_dashboard.png")

# Create a detailed summary report
print("\n" + "="*100)
print("AGI FORECAST FRAMEWORK - SUMMARY REPORT")
print("="*100)

print(f"\n📊 LUCR (Loss to Utility Conversion Rate) Analysis:")
print(f"   Formula: LUCR = Kaplan Prediction - Actual Capability")
print(f"   ")
print(f"   Current LUCR: {current_lucr:+.2f}%")
print(f"   {'   → Positive LUCR = INEFFICIENT (burning compute without proportional gains)' if current_lucr > 0 else '   → Negative LUCR = EFFICIENT (algorithmic breakthroughs)'}")
print(f"   ")
print(f"   LUCR Trend: {z[0]:.4f}%/month")
print(f"   {'   → Efficiency is DECLINING (getting worse over time)' if z[0] > 0 else '   → Efficiency is IMPROVING (getting better over time)'}")

print(f"\n🔬 Scaling Law:")
print(f"   Type: Logarithmic")
print(f"   Function: score = 0.0145 * log10(compute) - 0.303")
print(f"   R² = 0.75")

print(f"\n🚀 Frontier Compute:")
print(f"   2019: {valid_data['max_compute_flops'].iloc[0]:.2e} FLOPs")
print(f"   2024: {current_compute:.2e} FLOPs")
print(f"   Growth: {growth_factor:.0f}x")

print(f"\n🎯 AGI Progress:")
print(f"   Actual Capability: {current_actual:.2f}%")
print(f"   Kaplan Prediction: {current_predicted:.2f}%")
print(f"   Gap: {abs(current_lucr):.1f}% {'below' if current_lucr > 0 else 'above'} prediction")

print(f"\n💡 KEY INSIGHT:")
if current_lucr > 0 and z[0] > 0:
    print(f"   ⚠️  CONCERNING: We're inefficient AND getting worse over time.")
    print(f"   → This challenges optimistic AGI timelines.")
    print(f"   → Raw compute scaling alone won't get us to AGI.")
    print(f"   → Need algorithmic/architectural breakthroughs.")
elif current_lucr < 0 and z[0] < 0:
    print(f"   ✅ ENCOURAGING: We're efficient AND improving over time.")
    print(f"   → Algorithmic progress is strong.")
    print(f"   → AGI timelines could be faster than compute alone predicts.")
elif current_lucr > 0 and z[0] < 0:
    print(f"   📊 MIXED: Currently inefficient but improving.")
    print(f"   → We're making progress on efficiency.")
    print(f"   → With continued improvement, could reach efficiency parity.")
else:
    print(f"   📊 MIXED: Currently efficient but declining.")
    print(f"   → We had early wins but hitting diminishing returns.")
    print(f"   → May need new breakthroughs to maintain efficiency.")

print("\n" + "="*100)


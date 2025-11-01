"""
STEP 7: GPT-2 Sensitivity Analysis

Investigates the impact of GPT-2 on the model fits:
1. Leave-one-out check (with vs without GPT-2)
2. Era stratification (pre-RLHF vs post-RLHF)
3. Compute threshold analysis (≥10²³ FLOPs)
4. Compare forecasts with different regimes

Output:
- Sensitivity analysis showing impact on slopes
- Comparison of forecasts with/without GPT-2
- Recommended regime for primary fit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy.special import expit, logit

print("="*100)
print("STEP 7: GPT-2 SENSITIVITY ANALYSIS")
print("="*100)

# ============================================================================
# Load data
# ============================================================================

stage1 = pd.read_csv('outputs/stage1_compute_eci.csv')
stage2 = pd.read_csv('outputs/stage2_eci_metr.csv')
metr_cap = pd.read_csv('outputs/metr_capability_summary.csv')

print(f"\nStage 1 (C→ECI): {len(stage1)} models")
print(f"  Compute range: {stage1['compute'].min():.2e} to {stage1['compute'].max():.2e}")

# ============================================================================
# Identify GPT-2 and compute thresholds
# ============================================================================

gpt2_compute = 1.5e21
threshold_1e23 = 1e23
threshold_1e24 = 1e24

gpt2_mask = stage1['compute'] < 1e22  # GPT-2 era
print(f"\nGPT-2 era models (< 10²² FLOPs): {gpt2_mask.sum()}")
print(stage1[gpt2_mask][['Model', 'compute', 'eci']])

# ============================================================================
# Stage 1: Leave-one-out analysis
# ============================================================================

print(f"\n{'='*100}")
print("STAGE 1: LEAVE-ONE-OUT ANALYSIS (Compute → ECI)")
print("="*100)

# Full fit
log_C_full = np.log10(stage1['compute'])
ECI_full = stage1['eci']
slope_full, intercept_full, r_full, p_full, stderr_full = linregress(log_C_full, ECI_full)

print(f"\nFull fit (all {len(stage1)} models):")
print(f"  ECI = {intercept_full:.2f} + {slope_full:.2f} * log₁₀(C)")
print(f"  R² = {r_full**2:.4f}")
print(f"  Slope SE = {stderr_full:.2f}")

# Without GPT-2 era
stage1_no_gpt2 = stage1[~gpt2_mask].copy()
log_C_no_gpt2 = np.log10(stage1_no_gpt2['compute'])
ECI_no_gpt2 = stage1_no_gpt2['eci']
slope_no_gpt2, intercept_no_gpt2, r_no_gpt2, p_no_gpt2, stderr_no_gpt2 = linregress(log_C_no_gpt2, ECI_no_gpt2)

print(f"\nWithout GPT-2 era ({len(stage1_no_gpt2)} models):")
print(f"  ECI = {intercept_no_gpt2:.2f} + {slope_no_gpt2:.2f} * log₁₀(C)")
print(f"  R² = {r_no_gpt2**2:.4f}")
print(f"  Slope SE = {stderr_no_gpt2:.2f}")

# Calculate impact
slope_change = slope_no_gpt2 - slope_full
slope_change_pct = (slope_change / slope_full) * 100
slope_change_sigma = slope_change / stderr_full

print(f"\n📊 Impact of excluding GPT-2:")
print(f"  Δ slope: {slope_change:+.3f} ({slope_change_pct:+.1f}%)")
print(f"  Δ slope in σ: {slope_change_sigma:+.2f}σ")
print(f"  Δ intercept: {intercept_no_gpt2 - intercept_full:+.2f}")

if abs(slope_change_pct) > 10:
    print(f"\n⚠️  SIGNIFICANT IMPACT: Slope changes by {abs(slope_change_pct):.1f}% (>10% threshold)")
    print(f"  → Recommend excluding GPT-2 era from primary fit")
else:
    print(f"\n✓ Minor impact: Slope changes by {abs(slope_change_pct):.1f}% (<10% threshold)")
    print(f"  → Can keep GPT-2 era in fit with down-weighting")

# ============================================================================
# Compute threshold analysis
# ============================================================================

print(f"\n{'='*100}")
print("COMPUTE THRESHOLD ANALYSIS")
print("="*100)

thresholds = [
    ('Full', stage1['compute'] > 0, len(stage1)),
    ('≥ 10²³', stage1['compute'] >= threshold_1e23, (stage1['compute'] >= threshold_1e23).sum()),
    ('≥ 10²⁴', stage1['compute'] >= threshold_1e24, (stage1['compute'] >= threshold_1e24).sum()),
    ('No GPT-2 era', ~gpt2_mask, (~gpt2_mask).sum())
]

results = []
for name, mask, n in thresholds:
    data = stage1[mask]
    log_C = np.log10(data['compute'])
    ECI = data['eci']
    
    if len(data) > 5:
        slope, intercept, r, p, stderr = linregress(log_C, ECI)
        results.append({
            'threshold': name,
            'n_models': n,
            'slope': slope,
            'intercept': intercept,
            'r2': r**2,
            'stderr': stderr
        })
        
        print(f"\n{name} ({n} models):")
        print(f"  ECI = {intercept:.2f} + {slope:.2f} * log₁₀(C)")
        print(f"  R² = {r**2:.4f}, SE = {stderr:.2f}")

results_df = pd.DataFrame(results)

# ============================================================================
# Era stratification analysis
# ============================================================================

print(f"\n{'='*100}")
print("ERA STRATIFICATION")
print("="*100)

# Define eras by compute
stage1['era'] = pd.cut(stage1['compute'], 
                       bins=[0, 1e22, 1e24, 1e30],
                       labels=['Pre-RLHF (<10²²)', 'GPT-3 era (10²²-10²⁴)', 'Modern (≥10²⁴)'])

print("\nModels per era:")
print(stage1['era'].value_counts().sort_index())

for era in stage1['era'].unique():
    era_data = stage1[stage1['era'] == era]
    if len(era_data) > 5:
        log_C = np.log10(era_data['compute'])
        ECI = era_data['eci']
        slope, intercept, r, p, stderr = linregress(log_C, ECI)
        
        print(f"\n{era}:")
        print(f"  ECI = {intercept:.2f} + {slope:.2f} * log₁₀(C)")
        print(f"  R² = {r**2:.4f}")

# ============================================================================
# Stage 2: Impact check (GPT-2 not in METR overlap)
# ============================================================================

print(f"\n{'='*100}")
print("STAGE 2: GPT-2 IMPACT (ECI → METR)")
print("="*100)

print(f"\nStage 2 models:")
print(stage2[['metr_model', 'compute', 'eci', 'metr']])

gpt2_in_stage2 = 'gpt2' in stage2['metr_model'].values
print(f"\nGPT-2 in Stage 2: {gpt2_in_stage2}")

if gpt2_in_stage2:
    print(f"⚠️  GPT-2 found in METR overlap - excluding for Stage 2 fit")
    stage2_filtered = stage2[stage2['metr_model'] != 'gpt2'].copy()
else:
    print(f"✓ GPT-2 not in METR overlap - no impact on Stage 2")
    stage2_filtered = stage2.copy()

# ============================================================================
# Visualization
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('GPT-2 Sensitivity Analysis', fontsize=18, fontweight='bold')

# Plot 1: Full fit vs No GPT-2
ax = axes[0, 0]

# All data
ax.scatter(log_C_full, ECI_full, s=50, alpha=0.5, c='steelblue', 
           edgecolors='black', label='All models')

# Highlight GPT-2 era
gpt2_log_C = log_C_full[gpt2_mask]
gpt2_ECI = ECI_full[gpt2_mask]
ax.scatter(gpt2_log_C, gpt2_ECI, s=150, alpha=0.8, c='red', 
           edgecolors='black', linewidth=2, marker='s', label='GPT-2 era (excluded)')

# Fits
compute_range = np.linspace(log_C_full.min(), log_C_full.max(), 100)

ax.plot(compute_range, intercept_full + slope_full * compute_range, 
        'b-', linewidth=3, label=f'Full fit (slope={slope_full:.2f})')
ax.plot(compute_range, intercept_no_gpt2 + slope_no_gpt2 * compute_range, 
        'g--', linewidth=3, label=f'No GPT-2 (slope={slope_no_gpt2:.2f})')

ax.set_xlabel('log₁₀(Training Compute)', fontsize=12, fontweight='bold')
ax.set_ylabel('ECI Score', fontsize=12, fontweight='bold')
ax.set_title('Stage 1: Impact of GPT-2 Era', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Add text box with impact
textstr = f'Δ slope: {slope_change:+.3f} ({slope_change_pct:+.1f}%)\nΔ slope: {slope_change_sigma:+.2f}σ'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, fontweight='bold')

# Plot 2: Threshold comparison
ax = axes[0, 1]

x_pos = np.arange(len(results_df))
bars = ax.bar(x_pos, results_df['slope'], alpha=0.7, 
              color=['steelblue', 'green', 'orange', 'purple'],
              edgecolor='black', linewidth=1.5)

# Error bars
ax.errorbar(x_pos, results_df['slope'], yerr=results_df['stderr'], 
            fmt='none', ecolor='black', capsize=10, linewidth=2)

ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['threshold'], fontsize=10, rotation=20, ha='right')
ax.set_ylabel('Slope (dECI/d log C)', fontsize=12, fontweight='bold')
ax.set_title('Slope by Compute Threshold', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add n_models labels
for i, row in results_df.iterrows():
    ax.text(i, row['slope'] + row['stderr'] + 0.3, f"n={int(row['n_models'])}",
            ha='center', fontsize=9, fontweight='bold')

# Plot 3: Residuals
ax = axes[1, 0]

# Full fit residuals
residuals_full = ECI_full - (intercept_full + slope_full * log_C_full)
ax.scatter(log_C_full[~gpt2_mask], residuals_full[~gpt2_mask], 
           s=50, alpha=0.5, c='steelblue', label='Post-GPT-2')
ax.scatter(log_C_full[gpt2_mask], residuals_full[gpt2_mask], 
           s=150, alpha=0.8, c='red', marker='s', edgecolors='black', linewidth=2,
           label='GPT-2 era')

ax.axhline(0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('log₁₀(Training Compute)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual (ECI)', fontsize=12, fontweight='bold')
ax.set_title('Residuals: Full Fit', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Era-specific slopes
ax = axes[1, 1]

era_slopes = []
era_names = []
era_colors = ['red', 'orange', 'green']

for i, era in enumerate(stage1['era'].unique()):
    era_data = stage1[stage1['era'] == era]
    if len(era_data) > 5:
        log_C = np.log10(era_data['compute'])
        ECI = era_data['eci']
        slope, intercept, r, p, stderr = linregress(log_C, ECI)
        
        era_slopes.append(slope)
        era_names.append(str(era))

x_pos = np.arange(len(era_slopes))
ax.bar(x_pos, era_slopes, alpha=0.7, color=era_colors[:len(era_slopes)],
       edgecolor='black', linewidth=1.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(era_names, fontsize=10, rotation=20, ha='right')
ax.set_ylabel('Slope (dECI/d log C)', fontsize=12, fontweight='bold')
ax.set_title('Slope by Era', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/06_gpt2_sensitivity.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: outputs/06_gpt2_sensitivity.png")

# ============================================================================
# Save results
# ============================================================================

sensitivity_file = 'outputs/gpt2_sensitivity_results.csv'
results_df.to_csv(sensitivity_file, index=False)
print(f"✅ Saved: {sensitivity_file}")

# ============================================================================
# Recommendation
# ============================================================================

print(f"\n{'='*100}")
print("RECOMMENDATION")
print("="*100)

if abs(slope_change_pct) > 10:
    print(f"\n🎯 PRIMARY FIT: Exclude GPT-2 era (< 10²² FLOPs)")
    print(f"   Reason: Slope changes by {abs(slope_change_pct):.1f}% (>{10}% threshold)")
    print(f"   New slope: {slope_no_gpt2:.2f} ± {stderr_no_gpt2:.2f}")
    print(f"\n📝 REPORTING:")
    print(f"   • Show GPT-2 as greyed, unweighted point labeled 'pre-RLHF regime; excluded from fit'")
    print(f"   • Note: Including GPT-2 shifts slope by {slope_change_pct:+.1f}%")
    print(f"   • Justification: GPT-2 represents different training paradigm (pre-RLHF)")
    
    recommended_slope = slope_no_gpt2
    recommended_intercept = intercept_no_gpt2
    recommended_n = len(stage1_no_gpt2)
else:
    print(f"\n✓ PRIMARY FIT: Keep all models (include GPT-2 with down-weighting)")
    print(f"   Reason: Slope changes by only {abs(slope_change_pct):.1f}% (<10% threshold)")
    print(f"   Slope: {slope_full:.2f} ± {stderr_full:.2f}")
    print(f"\n📝 REPORTING:")
    print(f"   • Include GPT-2 in primary fit")
    print(f"   • Show sensitivity: excluding GPT-2 changes slope by {slope_change_pct:+.1f}%")
    
    recommended_slope = slope_full
    recommended_intercept = intercept_full
    recommended_n = len(stage1)

print(f"\n{'='*100}")
print("✅ STEP 7 COMPLETE")
print("="*100)

print(f"\n💡 Recommended primary fit:")
print(f"  ECI = {recommended_intercept:.2f} + {recommended_slope:.2f} * log₁₀(C)")
print(f"  n = {recommended_n} models")


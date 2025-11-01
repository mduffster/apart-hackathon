#!/usr/bin/env python3
"""
LUCR Main Chart - Following structure of 46_tightened_bayesian_model.py
Uses current pipeline data with subhuman filtering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy.special import expit
from scipy.stats import linregress

print("="*100)
print("LUCR MAIN CHART (4-Panel Comprehensive)")
print("="*100)

# ============================================================================
# Load Data from Current Pipeline
# ============================================================================

posterior_df = pd.read_parquet('outputs/posterior_samples.parquet')
current_state = pd.read_csv('outputs/current_state.csv')
stage2_data = pd.read_csv('outputs/stage2_eci_metr.csv')

# Extract conditioned posterior samples
a_post = posterior_df['a'].values
b_post = posterior_df['b'].values
alpha_post = posterior_df['alpha'].values
beta_post = posterior_df['beta'].values

print(f"\nLoaded {len(a_post)} posterior samples")
print(f"Current model: {current_state['model'].iloc[0]}")
print(f"Current compute: {current_state['compute'].iloc[0]:.2e}")
print(f"Current METR: {current_state['metr_capability'].iloc[0]:.3f}")

# ============================================================================
# Get OLS for reference line
# ============================================================================

stage2_eci = stage2_data['eci'].values
stage2_metr = stage2_data['metr'].values

# Match original: logit transform for OLS
from scipy.special import logit
stage2_metr_bounded = np.clip(stage2_metr, 1e-6, 1-1e-6)
stage2_logit_metr = logit(stage2_metr_bounded)

slope_ols, intercept_ols, _, _, _ = linregress(stage2_eci, stage2_logit_metr)

print(f"\nOLS (Stage 2): β = {slope_ols:.4f}")
print(f"Posterior: β = {beta_post.mean():.4f} ± {beta_post.std():.4f}")

# ============================================================================
# Posterior Predictive Forecasts (Match original exactly)
# ============================================================================

compute_forecast = np.logspace(24, 28, 100)
log_compute_forecast = np.log10(compute_forecast)

metr_forecast_samples = []

for log_c in log_compute_forecast:
    eci_samples = a_post + b_post * log_c
    logit_metr_samples = alpha_post + beta_post * eci_samples
    metr_samples = expit(logit_metr_samples)
    metr_forecast_samples.append(metr_samples)

metr_forecast_samples = np.array(metr_forecast_samples)

metr_median = np.median(metr_forecast_samples, axis=1)
metr_p05 = np.percentile(metr_forecast_samples, 5, axis=1)
metr_p25 = np.percentile(metr_forecast_samples, 25, axis=1)
metr_p75 = np.percentile(metr_forecast_samples, 75, axis=1)
metr_p95 = np.percentile(metr_forecast_samples, 95, axis=1)

# ============================================================================
# LUCR Calculation (Match original exactly)
# ============================================================================

def compute_lucr_sample(log_compute, a, b, alpha, beta):
    """
    LUCR = (dECI/d log C) × (dMETR/dECI)
    
    dECI/d log C = b (slope of Stage 1)
    dMETR/dECI = β × p × (1-p) where p = expit(α + β × ECI)
    
    The (1-p) term causes LUCR to DECLINE as METR approaches 1.0
    """
    dECI_dlogC = b
    eci = a + b * log_compute
    logit_metr = alpha + beta * eci
    metr = expit(logit_metr)
    dMETR_dECI = beta * metr * (1 - metr)  # Derivative of sigmoid
    return dMETR_dECI * dECI_dlogC

lucr_samples = []
for log_c in log_compute_forecast:
    lucr_per_compute = []
    for i in range(len(a_post)):
        lucr_val = compute_lucr_sample(log_c, a_post[i], b_post[i], 
                                       alpha_post[i], beta_post[i])
        lucr_per_compute.append(lucr_val)
    lucr_samples.append(lucr_per_compute)

lucr_samples = np.array(lucr_samples)
lucr_median = np.median(lucr_samples, axis=1)
lucr_p05 = np.percentile(lucr_samples, 5, axis=1)
lucr_p95 = np.percentile(lucr_samples, 95, axis=1)

# Find peak LUCR
peak_idx = np.argmax(lucr_median)
peak_compute = compute_forecast[peak_idx]
peak_lucr = lucr_median[peak_idx]

print(f"\n{'='*100}")
print("LUCR ANALYSIS")
print("="*100)
print(f"\nPeak LUCR:")
print(f"  Location: {peak_compute:.2e} FLOPs")
print(f"  Median: {peak_lucr:.4f}")
print(f"  90% CI: [{lucr_p05[peak_idx]:.4f}, {lucr_p95[peak_idx]:.4f}]")

# Current LUCR
current_compute = current_state['compute'].iloc[0]
current_idx = np.argmin(np.abs(compute_forecast - current_compute))
current_lucr = lucr_median[current_idx]

print(f"\nCurrent LUCR (@ {current_compute:.2e}):")
print(f"  Median: {current_lucr:.4f}")

# ============================================================================
# CI Width Analysis (Match original)
# ============================================================================

compute_milestones = [1e25, 5e25, 1e26, 5e26, 1e27]
ci_widths_50 = []
ci_widths_90 = []

for c in compute_milestones:
    idx = np.argmin(np.abs(compute_forecast - c))
    ci_widths_50.append(metr_p75[idx] - metr_p25[idx])
    ci_widths_90.append(metr_p95[idx] - metr_p05[idx])

# ============================================================================
# 4-Panel Figure (Match original layout exactly)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# ----------------------------------------------------------------------------
# Panel 1 (Top Left): METR Forecast
# ----------------------------------------------------------------------------

ax1 = axes[0, 0]
ax1.fill_between(compute_forecast, metr_p05, metr_p95, alpha=0.2, color='blue', 
                 label='90% CI (Tightened)')
ax1.fill_between(compute_forecast, metr_p25, metr_p75, alpha=0.3, color='blue', 
                 label='50% CI (Tightened)')
ax1.plot(compute_forecast, metr_median, 'b-', linewidth=3, label='Median (Tightened)')

# Thresholds
ax1.axhline(0.8, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Near-AGI (0.8)')
ax1.axhline(0.9, color='red', linestyle='--', linewidth=2, alpha=0.5, label='AGI (0.9)')

# Current state
ax1.scatter([current_compute], [metr_median[current_idx]], s=200, c='red', 
           edgecolors='black', linewidth=2, zorder=10)

ax1.set_xscale('log')
ax1.set_xlabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax1.set_ylabel('METR Capability', fontsize=12, fontweight='bold')
ax1.set_title('Tightened Posterior: METR Forecast', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# ----------------------------------------------------------------------------
# Panel 2 (Top Right): LUCR
# ----------------------------------------------------------------------------

ax2 = axes[0, 1]
ax2.fill_between(compute_forecast, lucr_p05, lucr_p95, alpha=0.2, color='green', 
                 label='90% CI')
ax2.plot(compute_forecast, lucr_median, 'g-', linewidth=3, label='Median')
ax2.scatter([peak_compute], [peak_lucr], s=200, c='darkgreen', 
           edgecolors='black', linewidth=2, zorder=10, label=f'Peak @ {peak_compute:.2e}')

ax2.set_xscale('log')
ax2.set_xlabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax2.set_ylabel('LUCR (dMETR / d log₁₀ C)', fontsize=12, fontweight='bold')
ax2.set_title('Tightened Posterior: LUCR', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ----------------------------------------------------------------------------
# Panel 3 (Bottom Left): Posterior of β
# ----------------------------------------------------------------------------

ax3 = axes[1, 0]

# Plot histogram + KDE
from scipy.stats import gaussian_kde

ax3.hist(beta_post, bins=40, density=True, alpha=0.6, color='purple', edgecolor='black')

# Fit KDE
kde = gaussian_kde(beta_post)
x_range = np.linspace(beta_post.min(), beta_post.max(), 200)
ax3.plot(x_range, kde(x_range), 'purple', linewidth=3, label='Posterior PDF')

# Mark median and HDI
beta_median = np.median(beta_post)
beta_hdi_low = np.percentile(beta_post, 5)
beta_hdi_high = np.percentile(beta_post, 95)

ax3.axvline(beta_median, color='darkviolet', linestyle='-', linewidth=2, 
            label=f'median={beta_median:.3f}')
ax3.axvline(beta_hdi_low, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax3.axvline(beta_hdi_high, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
            label='90% HDI')
ax3.axvspan(beta_hdi_low, beta_hdi_high, alpha=0.2, color='purple')

# OLS reference
ax3.axvline(slope_ols, color='red', linestyle='--', linewidth=2, label=f'OLS: {slope_ols:.4f}')

ax3.set_xlabel('β (Translation Efficiency)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
ax3.set_title('Posterior: β (Translation Efficiency)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ----------------------------------------------------------------------------
# Panel 4 (Bottom Right): Uncertainty vs Compute
# ----------------------------------------------------------------------------

ax4 = axes[1, 1]

x_labels = [f'{c:.0e}' for c in compute_milestones]
x_pos = range(len(x_labels))

ax4.bar(x_pos, ci_widths_90, alpha=0.5, color='purple', label='90% CI width')
ax4.bar(x_pos, ci_widths_50, alpha=0.7, color='purple', label='50% CI width')

ax4.set_xticks(x_pos)
ax4.set_xticklabels(x_labels, rotation=45, ha='right')
ax4.set_xlabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Credible Interval Width (METR)', fontsize=12, fontweight='bold')
ax4.set_title('Uncertainty vs Compute', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Save
# ============================================================================

plt.tight_layout()
plt.savefig('outputs/09_lucr_main_chart.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: outputs/09_lucr_main_chart.png")
plt.close()

# Save forecast data
forecast_df = pd.DataFrame({
    'compute': compute_forecast,
    'metr_median': metr_median,
    'metr_p05': metr_p05,
    'metr_p25': metr_p25,
    'metr_p75': metr_p75,
    'metr_p95': metr_p95,
    'lucr_median': lucr_median,
    'lucr_p05': lucr_p05,
    'lucr_p95': lucr_p95
})
forecast_df.to_csv('outputs/lucr_forecast.csv', index=False)
print(f"✅ Saved: outputs/lucr_forecast.csv")

print(f"\n{'='*100}")
print("✅ LUCR MAIN CHART COMPLETE")
print("="*100)

print(f"\n💡 KEY INSIGHTS:")
print(f"  • LUCR peaks at {peak_compute:.2e} FLOPs then DECLINES")
print(f"  • Current LUCR: {current_lucr:.4f} (below peak)")
print(f"  • Translation efficiency β: {beta_post.mean():.4f} ± {beta_post.std():.4f}")
print(f"  • Scaling efficiency b: {b_post.mean():.2f} ± {b_post.std():.2f}")

"""
STEP 3: Bayesian Hierarchical Model (C → ECI → METR)

Fits a two-stage hierarchical Bayesian model:
- Stage 1: Compute → ECI (linear on log scale, ~80 models)
- Stage 2: ECI → METR (logit link, ~8 models)

With improvements:
- Informative priors on β (translation efficiency)
- Hierarchical pooling for Stage 2
- Constrained σ_METR
- Posterior conditioning on observed current METR

Output:
- Posterior samples for all parameters
- Credible intervals for forecasts
- LUCR (scaling and translation efficiency)
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)

print("="*100)
print("STEP 3: BAYESIAN HIERARCHICAL MODEL")
print("="*100)

# ============================================================================
# Load data
# ============================================================================

stage1_data = pd.read_csv('outputs/stage1_compute_eci.csv')
stage2_data = pd.read_csv('outputs/stage2_eci_metr.csv')
current_state = pd.read_csv('outputs/current_state.csv')

print(f"\nStage 1 (C→ECI): {len(stage1_data)} models")
print(f"Stage 2 (ECI→METR): {len(stage2_data)} models")

# Prepare arrays with proper clipping
compute_epoch = np.clip(stage1_data['compute'].values, 1e-30, None)
eci_epoch = stage1_data['eci'].values
log_compute_epoch = np.log10(compute_epoch)

# Define eras for stratification (based on GPT-2 sensitivity analysis)
# Pre-RLHF: <10^22, GPT-3 era: 10^22-10^24, Modern: >=10^24
era_labels = []
for c in compute_epoch:
    if c < 1e22:
        era_labels.append(0)  # Pre-RLHF
    elif c < 1e24:
        era_labels.append(1)  # GPT-3 era
    else:
        era_labels.append(2)  # Modern
era_idx = np.array(era_labels)

print(f"\nEra stratification:")
print(f"  Pre-RLHF (<10²²): {(era_idx == 0).sum()} models")
print(f"  GPT-3 era (10²²-10²⁴): {(era_idx == 1).sum()} models")  
print(f"  Modern (≥10²⁴): {(era_idx == 2).sum()} models")

eci_overlap = stage2_data['eci'].values
metr_overlap = stage2_data['metr'].values
metr_overlap_clipped = np.clip(metr_overlap, 1e-6, 1-1e-6)  # Proper clipping for logit

# Save overlap model list for reproducibility
overlap_models = {
    'models': stage2_data['metr_model'].tolist(),
    'compute': stage2_data['compute'].tolist(),
    'eci': stage2_data['eci'].tolist(),
    'metr': stage2_data['metr'].tolist()
}
import json
with open('outputs/overlap_models.json', 'w') as f:
    json.dump(overlap_models, f, indent=2)
print(f"\n✅ Saved overlap model list: outputs/overlap_models.json")

# Get OLS estimates for priors
from scipy.stats import linregress

# Stage 1 OLS
slope_1, intercept_1, r_value_1, _, _ = linregress(log_compute_epoch, eci_epoch)
print(f"\nStage 1 OLS: ECI = {intercept_1:.2f} + {slope_1:.2f} * log10(C), R²={r_value_1**2:.4f}")

# Stage 2 OLS (logit scale)
logit_metr = logit(metr_overlap_clipped)
slope_2, intercept_2, r_value_2, _, _ = linregress(eci_overlap, logit_metr)
print(f"Stage 2 OLS: logit(METR) = {intercept_2:.3f} + {slope_2:.4f} * ECI, R²={r_value_2**2:.4f}")

# ============================================================================
# Build hierarchical model with tightened priors
# ============================================================================

print(f"\n{'='*100}")
print("FITTING BAYESIAN MODEL")
print("="*100)

# Add 10 pseudo-observations for Stage 2 (hierarchical pooling)
n_pseudo = 10
eci_pseudo = np.linspace(eci_overlap.min(), eci_overlap.max(), n_pseudo)
metr_pseudo = expit(intercept_2 + slope_2 * eci_pseudo)  # From OLS
metr_pseudo_clipped = np.clip(metr_pseudo, 1e-6, 1-1e-6)

# Combine real + pseudo
eci_stage2 = np.concatenate([eci_overlap, eci_pseudo])
metr_stage2 = np.concatenate([metr_overlap_clipped, metr_pseudo_clipped])
logit_metr_stage2 = logit(metr_stage2)

print(f"\nStage 2 augmented data:")
print(f"  Real observations: {len(eci_overlap)}")
print(f"  Pseudo observations: {n_pseudo}")
print(f"  Total: {len(eci_stage2)}")

with pm.Model() as lucr_model:
    # ========================================================================
    # Stage 1: Compute → ECI (with era stratification)
    # ========================================================================
    
    # Global priors
    a = pm.Normal('a', 100, 40)  # Global intercept
    b = pm.Normal('b', 10, 5)    # Global slope
    sigma_eci = pm.HalfNormal('sigma_eci', 10)
    
    # Era-specific intercept adjustments (partial pooling)
    # γ_era ~ Normal(0, σ_era) with hierarchical prior on σ_era
    sigma_era = pm.HalfNormal('sigma_era', 10)
    gamma_era = pm.Normal('gamma_era', 0, sigma_era, shape=3)  # 3 eras
    
    # Model: ECI = a + b*logC + γ_era[era]
    mu_eci = a + b * log_compute_epoch + gamma_era[era_idx]
    
    # Likelihood: Student-t with fixed nu=5 for robustness to outliers
    pm.StudentT('ECI_obs', nu=5, mu=mu_eci, sigma=sigma_eci, observed=eci_epoch)
    
    # ========================================================================
    # Stage 2: ECI → METR (logit link)
    # ========================================================================
    
    # Informative priors centered on OLS
    alpha = pm.Normal('alpha', 0, 5)
    beta = pm.Normal('beta', 0.09, 0.03)  # Centered on empirical ~0.09
    
    # Constrained noise
    sigma_metr = pm.HalfNormal('sigma_metr', 0.1)
    
    # Likelihood: Student-t with fixed nu=5 (augmented with pseudo-observations)
    mu_logit = alpha + beta * eci_stage2
    pm.StudentT('logit_METR_obs', nu=5, mu=mu_logit, sigma=sigma_metr,
                observed=logit_metr_stage2)
    
    # Sample
    n_draws = 4000
    n_tune = 2000
    print(f"\nSampling posterior (4 chains × {n_draws} samples)...")
    trace = pm.sample(n_draws, tune=n_tune, chains=4, target_accept=0.95, 
                      random_seed=42, progressbar=True)

# ============================================================================
# Check convergence
# ============================================================================

print(f"\n{'='*100}")
print("CONVERGENCE DIAGNOSTICS")
print("="*100)

summary = az.summary(trace, hdi_prob=0.9)
print(summary)

print(f"\nConvergence:")
for var in ['a', 'b', 'alpha', 'beta']:
    rhat = summary.loc[var, 'r_hat']
    ess = summary.loc[var, 'ess_bulk']
    status = "✓" if rhat < 1.01 and ess > 1000 else "✗"
    print(f"  {var:8s}: R̂ = {rhat:.4f}, ESS = {ess:6.0f} {status}")

# Check era parameters
print(f"\nEra intercept adjustments (γ_era):")
for i in range(3):
    era_name = ['Pre-RLHF', 'GPT-3 era', 'Modern'][i]
    gamma_mean = summary.loc[f'gamma_era[{i}]', 'mean']
    gamma_hdi_low = summary.loc[f'gamma_era[{i}]', 'hdi_5%']
    gamma_hdi_high = summary.loc[f'gamma_era[{i}]', 'hdi_95%']
    print(f"  {era_name:12s}: γ = {gamma_mean:+.2f} [{gamma_hdi_low:+.2f}, {gamma_hdi_high:+.2f}]")

# ============================================================================
# Condition posterior on observed current METR
# ============================================================================

print(f"\n{'='*100}")
print("CONDITIONING POSTERIOR")
print("="*100)

# Extract raw samples
posterior = trace.posterior
a_post_raw = posterior['a'].values.flatten()
b_post_raw = posterior['b'].values.flatten()
alpha_post_raw = posterior['alpha'].values.flatten()
beta_post_raw = posterior['beta'].values.flatten()

# Current state
best_current = current_state.iloc[0]
conditioning_compute = best_current['compute']
conditioning_metr = best_current['metr_capability']
upper_bound = conditioning_metr + 0.07  # Allow some tolerance

print(f"\nObserved current state:")
print(f"  Model: {best_current['model']}")
print(f"  Compute: {conditioning_compute:.2e}")
print(f"  METR: {conditioning_metr:.3f}")
print(f"  Upper bound: {upper_bound:.2f}")

# Predict current METR for all posterior samples
log_curr = np.log10(conditioning_compute)
eci_curr = a_post_raw + b_post_raw * log_curr
logit_metr_curr = alpha_post_raw + beta_post_raw * eci_curr
metr_curr_samples = expit(logit_metr_curr)

# Reject samples predicting current METR > upper_bound
valid_mask = metr_curr_samples <= upper_bound
n_rejected = np.sum(~valid_mask)
rejection_rate = n_rejected / len(valid_mask)

print(f"\nRejecting {n_rejected}/{len(valid_mask)} samples ({rejection_rate*100:.1f}%)")

# Conditioned posterior
a_post = a_post_raw[valid_mask]
b_post = b_post_raw[valid_mask]
alpha_post = alpha_post_raw[valid_mask]
beta_post = beta_post_raw[valid_mask]

print(f"Remaining samples: {len(a_post)}")

# Verify
metr_curr_conditioned = metr_curr_samples[valid_mask]
print(f"\nConditioned posterior for current METR:")
print(f"  Median: {np.median(metr_curr_conditioned):.3f}")
print(f"  90% CI: [{np.percentile(metr_curr_conditioned, 5):.3f}, {np.percentile(metr_curr_conditioned, 95):.3f}]")
print(f"  (Observed: {conditioning_metr:.3f})")

# ============================================================================
# Posterior summary
# ============================================================================

print(f"\n{'='*100}")
print("POSTERIOR SUMMARY (CONDITIONED)")
print("="*100)

print(f"\nStage 1 (C → ECI):")
print(f"  a (intercept): {a_post.mean():.2f} ± {a_post.std():.2f}")
print(f"  b (slope): {b_post.mean():.2f} ± {b_post.std():.2f}")
print(f"  [OLS: a={intercept_1:.2f}, b={slope_1:.2f}]")

print(f"\nStage 2 (ECI → METR):")
print(f"  α (intercept): {alpha_post.mean():.3f} ± {alpha_post.std():.3f}")
print(f"  β (translation efficiency): {beta_post.mean():.4f} ± {beta_post.std():.4f}")
print(f"  [OLS: α={intercept_2:.3f}, β={slope_2:.4f}]")

# ============================================================================
# Calculate LUCR (scaling + translation efficiency)
# ============================================================================

print(f"\n{'='*100}")
print("LUCR DECOMPOSITION")
print("="*100)

# Peak LUCR location (where dECI/d(log C) is maximum)
# For linear model: constant, so use midpoint of observed range
peak_compute = np.median(compute_epoch)
log_peak = np.log10(peak_compute)

# Scaling efficiency: dECI / d(log C)
scaling_eff = b_post  # Constant for linear model

# Translation efficiency: dMETR / dECI (at median ECI)
median_eci = np.median(eci_epoch)
# For logit link: dMETR/dECI = β * METR * (1-METR)
metr_at_median = expit(alpha_post + beta_post * median_eci)
translation_eff = beta_post * metr_at_median * (1 - metr_at_median)

# Combined LUCR: d(METR) / d(log C)
lucr = scaling_eff * translation_eff

print(f"\nLUCR at median compute ({peak_compute:.2e} FLOPs):")
print(f"  Scaling efficiency (dECI/d(log C)): {np.median(scaling_eff):.2f} [{np.percentile(scaling_eff, 5):.2f}, {np.percentile(scaling_eff, 95):.2f}]")
print(f"  Translation efficiency (dMETR/dECI): {np.median(translation_eff):.4f} [{np.percentile(translation_eff, 5):.4f}, {np.percentile(translation_eff, 95):.4f}]")
print(f"  Combined LUCR (dMETR/d(log C)): {np.median(lucr):.3f} [{np.percentile(lucr, 5):.3f}, {np.percentile(lucr, 95):.3f}]")

# ============================================================================
# Posterior Predictive Checks (PPC)
# ============================================================================

print(f"\n{'='*100}")
print("POSTERIOR PREDICTIVE CHECKS")
print("="*100)

# Generate posterior predictive samples
with lucr_model:
    ppc_samples = pm.sample_posterior_predictive(trace, random_seed=42)

# Extract observed and simulated data
eci_obs = eci_epoch
eci_sim = ppc_samples.posterior_predictive['ECI_obs'].values  # shape: (chains, draws, n_models)

# For Stage 2, only use the real observations (not pseudo-data)
metr_obs_real = metr_overlap  # Original real observations only
logit_metr_sim = ppc_samples.posterior_predictive['logit_METR_obs'].values

# Flatten simulation arrays for percentile calculation
eci_sim_flat = eci_sim.reshape(-1, eci_sim.shape[-1])  # (n_samples, n_models)
logit_metr_sim_flat = logit_metr_sim.reshape(-1, logit_metr_sim.shape[-1])

# For Stage 2, only keep the first n_real observations (exclude pseudo-data)
n_real = len(metr_obs_real)
logit_metr_sim_real = logit_metr_sim_flat[:, :n_real]

# Compute HDI for Stage 1 (ECI)
eci_hdi_50_low = np.percentile(eci_sim_flat, 25, axis=0)
eci_hdi_50_high = np.percentile(eci_sim_flat, 75, axis=0)
eci_hdi_90_low = np.percentile(eci_sim_flat, 5, axis=0)
eci_hdi_90_high = np.percentile(eci_sim_flat, 95, axis=0)

# Compute HDI for Stage 2 (METR) - only for real observations
logit_metr_hdi_50_low = np.percentile(logit_metr_sim_real, 25, axis=0)
logit_metr_hdi_50_high = np.percentile(logit_metr_sim_real, 75, axis=0)
logit_metr_hdi_90_low = np.percentile(logit_metr_sim_real, 5, axis=0)
logit_metr_hdi_90_high = np.percentile(logit_metr_sim_real, 95, axis=0)

# Convert back to METR scale
from scipy.special import expit
metr_hdi_50_low = expit(logit_metr_hdi_50_low)
metr_hdi_50_high = expit(logit_metr_hdi_50_high)
metr_hdi_90_low = expit(logit_metr_hdi_90_low)
metr_hdi_90_high = expit(logit_metr_hdi_90_high)

# Create PPC plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Stage 1: Compute → ECI
# Sort for proper fill_between
sort_idx_1 = np.argsort(log_compute_epoch)
ax1.scatter(log_compute_epoch, eci_obs, s=80, alpha=0.8, label='Observed', zorder=3)
ax1.fill_between(log_compute_epoch[sort_idx_1], eci_hdi_90_low[sort_idx_1], eci_hdi_90_high[sort_idx_1], 
                  alpha=0.2, color='blue', label='90% HDI')
ax1.fill_between(log_compute_epoch[sort_idx_1], eci_hdi_50_low[sort_idx_1], eci_hdi_50_high[sort_idx_1], 
                  alpha=0.4, color='blue', label='50% HDI')
ax1.set_xlabel('log₁₀(Compute [FLOPs])', fontsize=12)
ax1.set_ylabel('ECI Score', fontsize=12)
ax1.set_title('Stage 1 PPC: Compute → ECI', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Stage 2: ECI → METR (only real observations)
# Sort for proper fill_between
sort_idx_2 = np.argsort(eci_overlap)
ax2.scatter(eci_overlap, metr_obs_real, s=80, alpha=0.8, label='Observed', zorder=3)
ax2.fill_between(eci_overlap[sort_idx_2], metr_hdi_90_low[sort_idx_2], metr_hdi_90_high[sort_idx_2],
                  alpha=0.2, color='green', label='90% HDI')
ax2.fill_between(eci_overlap[sort_idx_2], metr_hdi_50_low[sort_idx_2], metr_hdi_50_high[sort_idx_2],
                  alpha=0.4, color='green', label='50% HDI')
ax2.set_xlabel('ECI Score', fontsize=12)
ax2.set_ylabel('METR Capability', fontsize=12)
ax2.set_title('Stage 2 PPC: ECI → METR (Real Observations)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/08_posterior_predictive_checks.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved PPC plot: outputs/08_posterior_predictive_checks.png")
plt.close()

# Compute PPC statistics
print(f"\nStage 1 (Compute → ECI):")
print(f"  Observations within 50% HDI: {((eci_obs >= eci_hdi_50_low) & (eci_obs <= eci_hdi_50_high)).sum()}/{len(eci_obs)}")
print(f"  Observations within 90% HDI: {((eci_obs >= eci_hdi_90_low) & (eci_obs <= eci_hdi_90_high)).sum()}/{len(eci_obs)}")

print(f"\nStage 2 (ECI → METR) [Real observations only]:")
print(f"  Observations within 50% HDI: {((metr_obs_real >= metr_hdi_50_low) & (metr_obs_real <= metr_hdi_50_high)).sum()}/{len(metr_obs_real)}")
print(f"  Observations within 90% HDI: {((metr_obs_real >= metr_hdi_90_low) & (metr_obs_real <= metr_hdi_90_high)).sum()}/{len(metr_obs_real)}")

# ============================================================================
# Save posterior samples
# ============================================================================

posterior_df = pd.DataFrame({
    'a': a_post,
    'b': b_post,
    'alpha': alpha_post,
    'beta': beta_post
})

# Save as both CSV and Parquet (parquet is faster for large datasets)
posterior_file_csv = 'outputs/posterior_samples.csv'
posterior_file_parquet = 'outputs/posterior_samples.parquet'
posterior_df.to_csv(posterior_file_csv, index=False)
posterior_df.to_parquet(posterior_file_parquet, index=False)
print(f"\n✅ Saved: {posterior_file_csv}")
print(f"✅ Saved: {posterior_file_parquet} (for fast loading)")

# Save summary
summary_dict = {
    'parameter': ['a', 'b', 'alpha', 'beta', 'scaling_eff', 'translation_eff', 'lucr'],
    'mean': [a_post.mean(), b_post.mean(), alpha_post.mean(), beta_post.mean(),
             scaling_eff.mean(), translation_eff.mean(), lucr.mean()],
    'std': [a_post.std(), b_post.std(), alpha_post.std(), beta_post.std(),
            scaling_eff.std(), translation_eff.std(), lucr.std()],
    'p05': [np.percentile(a_post, 5), np.percentile(b_post, 5), np.percentile(alpha_post, 5), np.percentile(beta_post, 5),
            np.percentile(scaling_eff, 5), np.percentile(translation_eff, 5), np.percentile(lucr, 5)],
    'p95': [np.percentile(a_post, 95), np.percentile(b_post, 95), np.percentile(alpha_post, 95), np.percentile(beta_post, 95),
            np.percentile(scaling_eff, 95), np.percentile(translation_eff, 95), np.percentile(lucr, 95)]
}

summary_df = pd.DataFrame(summary_dict)
summary_file = 'outputs/posterior_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"✅ Saved: {summary_file}")

print(f"\n{'='*100}")
print("✅ STEP 3 COMPLETE")
print("="*100)

print(f"\n💡 SUMMARY:")
print(f"  • Fitted hierarchical model with {len(a_post)} conditioned samples")
print(f"  • Rejection rate: {rejection_rate*100:.1f}%")
print(f"  • Current METR: {np.median(metr_curr_conditioned):.3f} (observed: {conditioning_metr:.3f})")
print(f"  • Translation efficiency (β): {beta_post.mean():.4f} ± {beta_post.std():.4f}")
print(f"  • Combined LUCR: {lucr.mean():.3f} ± {lucr.std():.3f}")
print(f"\n📊 Next step: Run 04_scenario_forecasts.py to generate timelines")


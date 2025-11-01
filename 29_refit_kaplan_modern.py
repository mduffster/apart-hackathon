"""
Refit Kaplan Scaling Laws Using Modern Data (2023-2025)
Test if the ECI ceiling is real or an artifact of 2020 constants
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

print("="*100)
print("REFITTING KAPLAN WITH MODERN DATA")
print("="*100)

# Load ECI data
eci_data = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')
eci_data = eci_data.dropna(subset=['ECI Score', 'Training compute (FLOP)', 'Release date'])
eci_data['Release date'] = pd.to_datetime(eci_data['Release date'])

# Filter to recent models (2023+)
recent_cutoff = '2023-01-01'
recent_data = eci_data[eci_data['Release date'] >= recent_cutoff].copy()

print(f"\nOriginal data: {len(eci_data)} models")
print(f"Recent data (>= {recent_cutoff}): {len(recent_data)} models")

# Also look at 2024+ for comparison
recent_2024 = eci_data[eci_data['Release date'] >= '2024-01-01'].copy()
print(f"2024+ data: {len(recent_2024)} models")

# Old Kaplan constants (2020)
old_L_inf = 1.69
old_A = 1.69
old_alpha = 0.076

print(f"\n{'='*100}")
print("OLD KAPLAN (2020):")
print(f"{'='*100}")
print(f"  L(C) = {old_L_inf} + {old_A} * C^(-{old_alpha})")
print(f"  Asymptotic loss (L_inf): {old_L_inf}")

# Calculate what old Kaplan predicts for recent models
recent_data['old_kaplan_loss'] = old_L_inf + old_A * np.power(recent_data['Training compute (FLOP)'], -old_alpha)
print(f"\n  Recent models' predicted loss: {recent_data['old_kaplan_loss'].min():.4f} - {recent_data['old_kaplan_loss'].max():.4f}")
print(f"  Already at {(recent_data['old_kaplan_loss'].min() / old_L_inf * 100):.1f}% of L_inf!")

# Strategy: Fit new Kaplan to ECI directly
# Instead of fitting loss (which we don't have), fit: ECI = f(compute)
# Try multiple functional forms

print(f"\n{'='*100}")
print("FITTING NEW SCALING LAWS TO MODERN DATA:")
print(f"{'='*100}")

# Use recent data
X = recent_data['Training compute (FLOP)'].values
Y = recent_data['ECI Score'].values

print(f"\nData range:")
print(f"  Compute: {X.min():.2e} - {X.max():.2e} FLOPs")
print(f"  ECI: {Y.min():.1f} - {Y.max():.1f}")

# Test multiple functional forms
results = []

# 1. Power Law: ECI = a * C^b + c
def power_law(C, a, b, c):
    return a * np.power(C, b) + c

try:
    # Initial guess
    p0 = [1.0, 0.1, 50]
    popt_pow, pcov_pow = curve_fit(power_law, X, Y, p0=p0, maxfev=20000)
    
    y_pred = power_law(X, *popt_pow)
    r2 = 1 - (np.sum((Y - y_pred)**2) / np.sum((Y - Y.mean())**2))
    
    results.append({
        'name': 'Power Law',
        'formula': f'ECI = {popt_pow[0]:.2e} * C^{popt_pow[1]:.4f} + {popt_pow[2]:.2f}',
        'params': popt_pow,
        'r2': r2,
        'has_ceiling': popt_pow[1] < 0,  # Negative exponent means ceiling
        'func': power_law
    })
    
    print(f"\n✓ POWER LAW:")
    print(f"  ECI = {popt_pow[0]:.2e} * C^{popt_pow[1]:.4f} + {popt_pow[2]:.2f}")
    print(f"  R² = {r2:.4f}")
    if popt_pow[1] < 0:
        ceiling = popt_pow[2]
        print(f"  ⚠️  HAS CEILING at ECI {ceiling:.1f} (negative exponent)")
    else:
        print(f"  ✓ NO CEILING (positive exponent - unbounded growth)")
    
except Exception as e:
    print(f"\n✗ Power Law failed: {e}")

# 2. Logarithmic: ECI = a * log(C) + b
def logarithmic(C, a, b):
    return a * np.log(C) + b

try:
    popt_log, pcov_log = curve_fit(logarithmic, X, Y)
    
    y_pred = logarithmic(X, *popt_log)
    r2 = 1 - (np.sum((Y - y_pred)**2) / np.sum((Y - Y.mean())**2))
    
    results.append({
        'name': 'Logarithmic',
        'formula': f'ECI = {popt_log[0]:.2f} * ln(C) + {popt_log[1]:.2f}',
        'params': popt_log,
        'r2': r2,
        'has_ceiling': False,
        'func': logarithmic
    })
    
    print(f"\n✓ LOGARITHMIC:")
    print(f"  ECI = {popt_log[0]:.2f} * ln(C) + {popt_log[1]:.2f}")
    print(f"  R² = {r2:.4f}")
    print(f"  ✓ NO CEILING (log grows unbounded)")
    
except Exception as e:
    print(f"\n✗ Logarithmic failed: {e}")

# 3. Kaplan-style but fit to ECI: ECI_inf - A * C^(-alpha)
def kaplan_style(C, eci_inf, A, alpha):
    """Inverse of Kaplan - approaches ECI_inf from below"""
    return eci_inf - A * np.power(C, -alpha)

try:
    # Initial guess: ceiling at 200, large A, small alpha
    p0 = [200, 100, 0.05]
    popt_kap, pcov_kap = curve_fit(kaplan_style, X, Y, p0=p0, maxfev=20000,
                                    bounds=([150, 0, 0], [1000, 1000, 1.0]))
    
    y_pred = kaplan_style(X, *popt_kap)
    r2 = 1 - (np.sum((Y - y_pred)**2) / np.sum((Y - Y.mean())**2))
    
    results.append({
        'name': 'Kaplan-style',
        'formula': f'ECI = {popt_kap[0]:.1f} - {popt_kap[1]:.2f} * C^(-{popt_kap[2]:.4f})',
        'params': popt_kap,
        'r2': r2,
        'has_ceiling': True,
        'ceiling': popt_kap[0],
        'func': kaplan_style
    })
    
    print(f"\n✓ KAPLAN-STYLE:")
    print(f"  ECI = {popt_kap[0]:.1f} - {popt_kap[1]:.2f} * C^(-{popt_kap[2]:.4f})")
    print(f"  R² = {r2:.4f}")
    print(f"  ⚠️  HAS CEILING at ECI {popt_kap[0]:.1f}")
    
except Exception as e:
    print(f"\n✗ Kaplan-style failed: {e}")

# 4. Linear (in log space): log(ECI) vs log(C)
def log_linear(C, a, b):
    """ECI = exp(a * log(C) + b) = C^a * exp(b)"""
    return np.exp(a * np.log(C) + b)

try:
    # Fit in log space
    log_X = np.log(X)
    log_Y = np.log(Y)
    
    popt_ll, pcov_ll = np.polyfit(log_X, log_Y, 1, cov=True)
    a_ll, b_ll = popt_ll
    
    y_pred = log_linear(X, a_ll, b_ll)
    r2 = 1 - (np.sum((Y - y_pred)**2) / np.sum((Y - Y.mean())**2))
    
    results.append({
        'name': 'Log-Linear',
        'formula': f'ECI = C^{a_ll:.4f} * {np.exp(b_ll):.2e}',
        'params': [a_ll, b_ll],
        'r2': r2,
        'has_ceiling': a_ll <= 0,
        'func': log_linear
    })
    
    print(f"\n✓ LOG-LINEAR:")
    print(f"  ECI = C^{a_ll:.4f} * {np.exp(b_ll):.2e}")
    print(f"  R² = {r2:.4f}")
    if a_ll > 0:
        print(f"  ✓ NO CEILING (power law with positive exponent)")
    else:
        print(f"  ⚠️  HAS CEILING (non-positive exponent)")
    
except Exception as e:
    print(f"\n✗ Log-Linear failed: {e}")

# Find best fit
if results:
    best_fit = max(results, key=lambda x: x['r2'])
    
    print(f"\n{'='*100}")
    print(f"BEST FIT: {best_fit['name']} (R² = {best_fit['r2']:.4f})")
    print(f"{'='*100}")
    print(f"  Formula: {best_fit['formula']}")
    
    if best_fit['has_ceiling']:
        ceiling = best_fit.get('ceiling', best_fit['params'][0] if best_fit['name'] == 'Kaplan-style' else 'unknown')
        print(f"  ⚠️  Predicts ceiling at ECI {ceiling}")
    else:
        print(f"  ✓ Predicts unbounded growth")
    
    # Test extrapolation
    print(f"\n  Extrapolation tests:")
    test_computes = [1e27, 1e28, 1e30, 1e35, 1e40]
    for test_c in test_computes:
        try:
            pred_eci = best_fit['func'](test_c, *best_fit['params'])
            print(f"    {test_c:.0e} FLOPs → ECI {pred_eci:.1f}")
        except:
            print(f"    {test_c:.0e} FLOPs → Failed to predict")
    
    # Save best fit
    output = {
        'method': best_fit['name'],
        'formula': best_fit['formula'],
        'r2': best_fit['r2'],
        'has_ceiling': best_fit['has_ceiling'],
        'params': [float(p) for p in best_fit['params']],
        'data_source': f'ECI models from {recent_cutoff} onwards',
        'n_models': len(recent_data)
    }
    
    if best_fit['has_ceiling']:
        output['ceiling'] = float(best_fit.get('ceiling', best_fit['params'][0]))
    
    with open('outputs/modern_scaling_law.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to outputs/modern_scaling_law.json")
    
    # Visualize all fits
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extended compute range
    compute_extended = np.logspace(np.log10(X.min()), np.log10(X.max()) + 2, 500)
    
    for idx, result in enumerate(results[:4]):  # Max 4 plots
        ax = axes[idx // 2, idx % 2]
        
        # Data points
        ax.scatter(X, Y, s=100, alpha=0.7, color='#2E86AB',
                  edgecolors='black', linewidth=1.5, label='Recent Models (2023+)', zorder=10)
        
        # Fit line
        try:
            y_fit = result['func'](compute_extended, *result['params'])
            # Clip unrealistic values
            y_fit = np.clip(y_fit, 0, 1000)
            
            ax.plot(compute_extended, y_fit, '-', linewidth=3, color='#06A77D',
                   label='Fitted Law', alpha=0.8)
            
            # Mark ceiling if exists
            if result['has_ceiling'] and 'ceiling' in result:
                ax.axhline(result['ceiling'], color='red', linestyle='--',
                          linewidth=2, alpha=0.7, label=f'Ceiling ({result["ceiling"]:.0f})')
        except:
            pass
        
        ax.set_xscale('log')
        ax.set_xlabel('Training Compute (FLOPs)', fontsize=11, fontweight='bold')
        ax.set_ylabel('ECI Score', fontsize=11, fontweight='bold')
        ax.set_title(f"{result['name']} (R²={result['r2']:.3f})", 
                    fontsize=12, fontweight='bold', pad=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim([Y.min() - 20, min(Y.max() + 100, 500)])
    
    plt.tight_layout()
    plt.savefig('outputs/modern_scaling_laws_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to outputs/modern_scaling_laws_comparison.png")

else:
    print("\n✗ No successful fits!")

print(f"\n{'='*100}")
print("✅ MODERN KAPLAN REFIT COMPLETE!")
print(f"{'='*100}")

print(f"\n💡 NEXT STEP:")
print(f"   Use the best-fit scaling law to rebuild AGI forecast")
print(f"   Check if ceiling is higher/removed with modern data")


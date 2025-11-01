"""
Test Non-Linear Functional Forms for ECI → METR
The linear fit breaks down at extrapolation - test sigmoid, log, power law
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

print("="*100)
print("TESTING NON-LINEAR ECI → METR RELATIONSHIPS")
print("="*100)

# Load comparison data (EQUAL weighting)
comparison = pd.read_csv('outputs/metr_vs_eci_comparison.csv')
comparison = comparison.dropna(subset=['eci_score', 'metr_score_simple']).copy()

X = comparison['eci_score'].values
Y = comparison['metr_score_simple'].values  # EQUAL weighting

print(f"\nData points: {len(comparison)}")
print(f"ECI range: {X.min():.1f} - {X.max():.1f}")
print(f"METR range: {Y.min()*100:.1f}% - {Y.max()*100:.1f}%")

# Define functional forms
def linear(x, a, b):
    return a * x + b

def sigmoid(x, L, k, x0, b):
    """Sigmoid with floor"""
    return L / (1 + np.exp(-k * (x - x0))) + b

def logarithmic(x, a, b):
    return a * np.log(x) + b

def power_law(x, a, b, c):
    return a * (x ** b) + c

def sqrt_form(x, a, b):
    """Square root (sublinear)"""
    return a * np.sqrt(x) + b

# Fit all forms
results = []

# 1. Linear (baseline)
try:
    popt_lin, _ = curve_fit(linear, X, Y)
    y_pred_lin = linear(X, *popt_lin)
    ss_res = np.sum((Y - y_pred_lin) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2_lin = 1 - (ss_res / ss_tot)
    
    results.append({
        'form': 'Linear',
        'params': popt_lin,
        'r2': r2_lin,
        'func': linear
    })
    print(f"\n✓ Linear: R² = {r2_lin:.4f}")
    print(f"  METR = {popt_lin[0]:.6f} * ECI + {popt_lin[1]:.6f}")
except Exception as e:
    print(f"\n✗ Linear failed: {e}")

# 2. Sigmoid
try:
    # Initial guess: L=1 (max METR), k=0.1, x0=135, b=0
    p0_sig = [1.0, 0.1, 135, 0.0]
    popt_sig, _ = curve_fit(sigmoid, X, Y, p0=p0_sig, maxfev=10000)
    y_pred_sig = sigmoid(X, *popt_sig)
    ss_res = np.sum((Y - y_pred_sig) ** 2)
    r2_sig = 1 - (ss_res / ss_tot)
    
    results.append({
        'form': 'Sigmoid',
        'params': popt_sig,
        'r2': r2_sig,
        'func': sigmoid
    })
    print(f"\n✓ Sigmoid: R² = {r2_sig:.4f}")
    print(f"  METR = {popt_sig[0]:.3f} / (1 + exp(-{popt_sig[1]:.3f}*(ECI - {popt_sig[2]:.1f}))) + {popt_sig[3]:.3f}")
    print(f"  Asymptote: {popt_sig[0] + popt_sig[3]:.3f} ({(popt_sig[0] + popt_sig[3])*100:.1f}%)")
except Exception as e:
    print(f"\n✗ Sigmoid failed: {e}")

# 3. Logarithmic
try:
    popt_log, _ = curve_fit(logarithmic, X, Y)
    y_pred_log = logarithmic(X, *popt_log)
    ss_res = np.sum((Y - y_pred_log) ** 2)
    r2_log = 1 - (ss_res / ss_tot)
    
    results.append({
        'form': 'Logarithmic',
        'params': popt_log,
        'r2': r2_log,
        'func': logarithmic
    })
    print(f"\n✓ Logarithmic: R² = {r2_log:.4f}")
    print(f"  METR = {popt_log[0]:.6f} * ln(ECI) + {popt_log[1]:.6f}")
except Exception as e:
    print(f"\n✗ Logarithmic failed: {e}")

# 4. Power law
try:
    p0_pow = [0.1, 2.0, -10.0]
    popt_pow, _ = curve_fit(power_law, X, Y, p0=p0_pow, maxfev=10000)
    y_pred_pow = power_law(X, *popt_pow)
    ss_res = np.sum((Y - y_pred_pow) ** 2)
    r2_pow = 1 - (ss_res / ss_tot)
    
    results.append({
        'form': 'Power Law',
        'params': popt_pow,
        'r2': r2_pow,
        'func': power_law
    })
    print(f"\n✓ Power Law: R² = {r2_pow:.4f}")
    print(f"  METR = {popt_pow[0]:.6f} * ECI^{popt_pow[1]:.3f} + {popt_pow[2]:.3f}")
except Exception as e:
    print(f"\n✗ Power Law failed: {e}")

# 5. Square root (sublinear)
try:
    popt_sqrt, _ = curve_fit(sqrt_form, X, Y)
    y_pred_sqrt = sqrt_form(X, *popt_sqrt)
    ss_res = np.sum((Y - y_pred_sqrt) ** 2)
    r2_sqrt = 1 - (ss_res / ss_tot)
    
    results.append({
        'form': 'Square Root',
        'params': popt_sqrt,
        'r2': r2_sqrt,
        'func': sqrt_form
    })
    print(f"\n✓ Square Root: R² = {r2_sqrt:.4f}")
    print(f"  METR = {popt_sqrt[0]:.6f} * sqrt(ECI) + {popt_sqrt[1]:.6f}")
except Exception as e:
    print(f"\n✗ Square Root failed: {e}")

# Sort by R²
results_df = pd.DataFrame([{
    'form': r['form'],
    'r2': r['r2']
} for r in results]).sort_values('r2', ascending=False)

print(f"\n{'='*100}")
print("RANKING BY FIT QUALITY:")
print(f"{'='*100}")
print(results_df.to_string(index=False))

# Test extrapolation to ECI 150
eci_150 = 150.0
target_metr = 0.9

print(f"\n{'='*100}")
print(f"EXTRAPOLATION TEST: What METR at ECI {eci_150}?")
print(f"{'='*100}")

for r in results:
    try:
        pred_metr = r['func'](eci_150, *r['params'])
        pred_metr_pct = pred_metr * 100
        
        # How far from AGI?
        gap_to_agi = (target_metr - pred_metr) * 100
        
        print(f"\n{r['form']:15s}: METR = {pred_metr_pct:5.1f}% (Gap to AGI: {gap_to_agi:+5.1f} pp)")
        
        # Find ECI needed for METR = 0.9
        if r['form'] == 'Linear':
            eci_needed = (target_metr - r['params'][1]) / r['params'][0]
            print(f"                 Need ECI {eci_needed:.1f} for AGI")
        elif r['form'] == 'Sigmoid':
            # Solve sigmoid for target_metr
            L, k, x0, b = r['params']
            if target_metr > b and target_metr < (L + b):
                eci_needed = x0 - (1/k) * np.log((L / (target_metr - b)) - 1)
                print(f"                 Need ECI {eci_needed:.1f} for AGI")
            else:
                print(f"                 AGI unreachable (asymptote at {(L+b)*100:.1f}%)")
        elif r['form'] == 'Logarithmic':
            eci_needed = np.exp((target_metr - r['params'][1]) / r['params'][0])
            print(f"                 Need ECI {eci_needed:.1f} for AGI")
        
    except Exception as e:
        print(f"\n{r['form']:15s}: Extrapolation failed ({e})")

# Visualize all fits
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Extended range for visualization
eci_extended = np.linspace(120, 200, 500)

for idx, r in enumerate(results):
    ax = axes[idx]
    
    # Data points
    ax.scatter(X, Y * 100, s=150, alpha=0.7, color='#2E86AB', 
              edgecolors='black', linewidth=2, label='Data', zorder=10)
    
    # Fit on training range
    eci_fit = np.linspace(X.min(), X.max(), 100)
    y_fit = r['func'](eci_fit, *r['params']) * 100
    ax.plot(eci_fit, y_fit, '-', linewidth=3, color='#06A77D', label='Fit')
    
    # Extrapolation
    eci_extrap = np.linspace(X.max(), 200, 100)
    y_extrap = r['func'](eci_extrap, *r['params']) * 100
    ax.plot(eci_extrap, y_extrap, '--', linewidth=3, color='#F18F01', 
           alpha=0.7, label='Extrapolation')
    
    # AGI target
    ax.axhline(90, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='AGI (90%)')
    ax.axvline(150, color='red', linestyle='--', linewidth=2, alpha=0.5, label='ECI 150')
    
    # Mark ECI 150 prediction
    pred_150 = r['func'](150, *r['params']) * 100
    ax.scatter([150], [pred_150], s=300, marker='*', color='gold',
              edgecolors='black', linewidth=2, zorder=15)
    
    ax.set_xlabel('ECI Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('METR Score (%)', fontsize=11, fontweight='bold')
    ax.set_title(f"{r['form']} (R²={r['r2']:.3f})\nAt ECI 150: {pred_150:.1f}%", 
                fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([120, 200])
    ax.set_ylim([0, 120])

# Hide extra subplot
if len(results) < 6:
    axes[-1].axis('off')

plt.tight_layout()
plt.savefig('outputs/eci_metr_nonlinear_fits.png', dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to outputs/eci_metr_nonlinear_fits.png")

# Save best fit
best_fit = max(results, key=lambda x: x['r2'])
print(f"\n{'='*100}")
print(f"BEST FIT: {best_fit['form']} (R² = {best_fit['r2']:.4f})")
print(f"{'='*100}")

# Save results
output = {
    'best_form': best_fit['form'],
    'best_r2': best_fit['r2'],
    'best_params': best_fit['params'].tolist(),
    'all_results': [{
        'form': r['form'],
        'r2': r['r2'],
        'params': r['params'].tolist()
    } for r in results]
}

with open('outputs/eci_metr_nonlinear_fits.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💡 RECOMMENDATION:")
print(f"   Use {best_fit['form']} fit to avoid over-optimistic extrapolation")
print(f"   This respects diminishing returns at higher ECI")


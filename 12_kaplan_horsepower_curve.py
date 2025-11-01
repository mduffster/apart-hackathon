"""
Build Kaplan "Horsepower" Curve
Path: Compute → Loss (Kaplan) → ECI Capability → METR Capability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

# Load ECI data
print("Loading ECI data with compute...")
eci = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')

# Filter to models with both ECI score and compute
eci_complete = eci[
    eci['ECI Score'].notna() & 
    eci['Training compute (FLOP)'].notna()
].copy()

print(f"Models with ECI score and compute: {len(eci_complete)}")

# Step 1: Kaplan loss scaling
# L(C) = L_∞ + A * C^(-α)
L_inf = 1.69
A = 1.69
alpha = 0.076

def kaplan_loss(compute):
    """Predict cross-entropy loss from training compute"""
    return L_inf + A * np.power(compute, -alpha)

# Calculate predicted loss for each model
eci_complete['kaplan_loss'] = eci_complete['Training compute (FLOP)'].apply(kaplan_loss)

print("\nKaplan loss predictions:")
print(f"  Loss range: {eci_complete['kaplan_loss'].min():.4f} - {eci_complete['kaplan_loss'].max():.4f} nats")

# Step 2: Fit sigmoid to map loss → ECI capability
# K(L) = 1 / (1 + exp(-k * (L_0 - L)))
# But for ECI, we need to scale: ECI(L) = ECI_min + (ECI_max - ECI_min) * sigmoid(L)

def sigmoid_loss_to_capability(loss, k, L_0, ECI_min, ECI_max):
    """Map loss to ECI capability using sigmoid"""
    # Note: Lower loss = higher capability, so we use (L_0 - L)
    sigmoid = 1.0 / (1.0 + np.exp(-k * (L_0 - loss)))
    return ECI_min + (ECI_max - ECI_min) * sigmoid

# Fit the sigmoid
X = eci_complete['kaplan_loss'].values
y = eci_complete['ECI Score'].values

# Initial guess
k_init = 10  # Steepness
L_0_init = X.mean()  # Midpoint
ECI_min_init = y.min()
ECI_max_init = y.max()

try:
    popt, pcov = curve_fit(
        sigmoid_loss_to_capability, 
        X, y,
        p0=[k_init, L_0_init, ECI_min_init, ECI_max_init],
        maxfev=10000
    )
    k_fit, L_0_fit, ECI_min_fit, ECI_max_fit = popt
    
    # Calculate R²
    y_pred = sigmoid_loss_to_capability(X, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n✓ Loss → ECI sigmoid fit:")
    print(f"  k (steepness): {k_fit:.4f}")
    print(f"  L_0 (midpoint): {L_0_fit:.4f} nats")
    print(f"  ECI range: {ECI_min_fit:.2f} - {ECI_max_fit:.2f}")
    print(f"  R² = {r2:.4f}")
    
except Exception as e:
    print(f"\n✗ Sigmoid fit failed: {e}")
    print("Using linear approximation instead...")
    # Fallback to linear
    coeffs = np.polyfit(X, y, 1)
    k_fit, L_0_fit, ECI_min_fit, ECI_max_fit = None, None, None, None

# Step 3: Define full Kaplan horsepower function
def kaplan_horsepower_eci(compute):
    """Predict ECI capability from compute using Kaplan scaling"""
    loss = kaplan_loss(compute)
    if k_fit is not None:
        eci = sigmoid_loss_to_capability(loss, k_fit, L_0_fit, ECI_min_fit, ECI_max_fit)
    else:
        # Linear fallback
        eci = coeffs[0] * loss + coeffs[1]
    return eci

# Step 4: Convert to METR using our mapping
# METR = 0.0017 * ECI - 0.208
eci_to_metr_slope = 0.001745
eci_to_metr_intercept = -0.208016

def kaplan_horsepower_metr(compute):
    """Predict METR capability from compute using Kaplan scaling"""
    eci = kaplan_horsepower_eci(compute)
    metr = eci_to_metr_slope * eci + eci_to_metr_intercept
    return metr

# Test the full pipeline
print("\n" + "="*100)
print("KAPLAN HORSEPOWER PREDICTIONS:")
print("="*100)

test_computes = [1e23, 1e24, 1e25, 1e26, 1e27]
for compute in test_computes:
    loss = kaplan_loss(compute)
    eci = kaplan_horsepower_eci(compute)
    metr = kaplan_horsepower_metr(compute)
    print(f"  {compute:.1e} FLOPs → Loss: {loss:.4f} → ECI: {eci:.1f} → METR: {metr:.4f}")

# Save horsepower function parameters
horsepower_params = {
    'kaplan_constants': {
        'L_inf': L_inf,
        'A': A,
        'alpha': alpha
    },
    'sigmoid_fit': {
        'k': float(k_fit) if k_fit is not None else None,
        'L_0': float(L_0_fit) if L_0_fit is not None else None,
        'ECI_min': float(ECI_min_fit) if ECI_min_fit is not None else None,
        'ECI_max': float(ECI_max_fit) if ECI_max_fit is not None else None,
        'r2': float(r2) if k_fit is not None else None
    },
    'eci_to_metr': {
        'slope': eci_to_metr_slope,
        'intercept': eci_to_metr_intercept
    }
}

with open('outputs/kaplan_horsepower_params.json', 'w') as f:
    json.dump(horsepower_params, f, indent=2)

print(f"\nSaved horsepower parameters to outputs/kaplan_horsepower_params.json")

# Visualize the pipeline
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Compute → Loss (Kaplan)
ax1 = axes[0]
compute_range = np.logspace(20, 27, 200)
loss_range = kaplan_loss(compute_range)

ax1.plot(compute_range, loss_range, linewidth=3, color='#2E86AB')
ax1.scatter(eci_complete['Training compute (FLOP)'], eci_complete['kaplan_loss'],
           s=100, alpha=0.5, edgecolors='black', linewidth=1)

ax1.set_xscale('log')
ax1.set_xlabel('Training Compute (FLOPs)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Cross-Entropy Loss (nats)', fontsize=11, fontweight='bold')
ax1.set_title('Kaplan Scaling: Compute → Loss', fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3)

# Plot 2: Loss → ECI (Sigmoid)
ax2 = axes[1]
loss_smooth = np.linspace(X.min(), X.max(), 200)
if k_fit is not None:
    eci_smooth = sigmoid_loss_to_capability(loss_smooth, k_fit, L_0_fit, ECI_min_fit, ECI_max_fit)
else:
    eci_smooth = coeffs[0] * loss_smooth + coeffs[1]

ax2.plot(loss_smooth, eci_smooth, linewidth=3, color='#F18F01')
ax2.scatter(X, y, s=100, alpha=0.5, edgecolors='black', linewidth=1)

ax2.set_xlabel('Cross-Entropy Loss (nats)', fontsize=11, fontweight='bold')
ax2.set_ylabel('ECI Score', fontsize=11, fontweight='bold')
ax2.set_title('Loss → Capability (Sigmoid)', fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()  # Lower loss is better

# Plot 3: Compute → METR (Full Pipeline)
ax3 = axes[2]
metr_range = kaplan_horsepower_metr(compute_range)

ax3.plot(compute_range, metr_range, linewidth=3, color='#E63946')
ax3.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Human-level (1.0)')
ax3.axhline(0.9, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Near-AGI (0.9)')

ax3.set_xscale('log')
ax3.set_xlabel('Training Compute (FLOPs)', fontsize=11, fontweight='bold')
ax3.set_ylabel('METR Frontier Score', fontsize=11, fontweight='bold')
ax3.set_title('Kaplan "Horsepower": Compute → METR', fontsize=13, fontweight='bold', pad=15)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, min(1.2, metr_range.max() * 1.1)])

plt.tight_layout()
plt.savefig('outputs/kaplan_horsepower_pipeline.png', dpi=300, bbox_inches='tight')
print("Saved pipeline visualization to outputs/kaplan_horsepower_pipeline.png")

print("\n" + "="*100)
print("✅ KAPLAN HORSEPOWER CURVE COMPLETE!")
print("="*100)
print("\nNext step: Calculate LUCR by comparing Kaplan predictions to actual METR frontier")


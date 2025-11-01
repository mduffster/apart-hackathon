"""
Rebuild Kaplan horsepower with LINEAR loss → ECI mapping
This prevents artificial saturation from sigmoid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Load ECI data
print("Loading ECI data...")
eci = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')

eci_complete = eci[
    eci['ECI Score'].notna() & 
    eci['Training compute (FLOP)'].notna()
].copy()

print(f"Models with ECI score and compute: {len(eci_complete)}")

# Step 1: Kaplan loss scaling (unchanged)
L_inf = 1.69
A = 1.69
alpha = 0.076

def kaplan_loss(compute):
    return L_inf + A * np.power(compute, -alpha)

# Calculate Kaplan loss for each model
eci_complete['kaplan_loss'] = eci_complete['Training compute (FLOP)'].apply(kaplan_loss)

print(f"\nKaplan loss range: {eci_complete['kaplan_loss'].min():.4f} - {eci_complete['kaplan_loss'].max():.4f} nats")

# Step 2: LINEAR fit loss → ECI
X = eci_complete['kaplan_loss'].values
y = eci_complete['ECI Score'].values

# Linear regression: ECI = a * loss + b
# Note: LOWER loss = HIGHER capability, so we expect negative slope
coeffs = np.polyfit(X, y, 1)
a_linear, b_linear = coeffs

print(f"\n✓ Linear fit: ECI = {a_linear:.4f} * loss + {b_linear:.4f}")

# Calculate R²
y_pred = a_linear * X + b_linear
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"  R² = {r2:.4f}")

# Step 3: Define full Kaplan horsepower function
def kaplan_horsepower_eci_linear(compute):
    """Predict ECI from compute using Kaplan with linear mapping"""
    loss = kaplan_loss(compute)
    eci = a_linear * loss + b_linear
    return eci

# Test predictions
print("\n" + "="*100)
print("KAPLAN HORSEPOWER PREDICTIONS (Linear):")
print("="*100)

test_computes = [1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e30]
for compute in test_computes:
    loss = kaplan_loss(compute)
    eci = kaplan_horsepower_eci_linear(compute)
    print(f"  {compute:.1e} FLOPs → Loss: {loss:.4f} → ECI: {eci:.1f}")

# Save parameters
horsepower_params_linear = {
    'kaplan_constants': {
        'L_inf': L_inf,
        'A': A,
        'alpha': alpha
    },
    'linear_fit': {
        'a': float(a_linear),
        'b': float(b_linear),
        'r2': float(r2)
    }
}

with open('outputs/kaplan_horsepower_linear.json', 'w') as f:
    json.dump(horsepower_params_linear, f, indent=2)

print(f"\nSaved linear parameters to outputs/kaplan_horsepower_linear.json")

# Visualize comparison: Sigmoid vs Linear
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Load sigmoid parameters for comparison
with open('outputs/kaplan_horsepower_params.json', 'r') as f:
    sigmoid_params = json.load(f)

k_fit = sigmoid_params['sigmoid_fit']['k']
L_0_fit = sigmoid_params['sigmoid_fit']['L_0']
ECI_min_fit = sigmoid_params['sigmoid_fit']['ECI_min']
ECI_max_fit = sigmoid_params['sigmoid_fit']['ECI_max']

def sigmoid_loss_to_eci(loss):
    sigmoid = 1.0 / (1.0 + np.exp(-k_fit * (L_0_fit - loss)))
    return ECI_min_fit + (ECI_max_fit - ECI_min_fit) * sigmoid

# Plot 1: Loss → ECI comparison
ax1 = axes[0]
loss_range = np.linspace(X.min() - 0.01, X.max() + 0.01, 200)

# Sigmoid
eci_sigmoid = [sigmoid_loss_to_eci(l) for l in loss_range]
ax1.plot(loss_range, eci_sigmoid, '--', linewidth=2.5, color='#E63946', 
        alpha=0.7, label='Sigmoid (saturates)')

# Linear
eci_linear = a_linear * loss_range + b_linear
ax1.plot(loss_range, eci_linear, '-', linewidth=2.5, color='#06A77D',
        alpha=0.7, label='Linear (extrapolates)')

# Data points
ax1.scatter(X, y, s=80, alpha=0.6, edgecolors='black', linewidth=1, color='#2E86AB', zorder=10)

ax1.set_xlabel('Cross-Entropy Loss (nats)', fontsize=11, fontweight='bold')
ax1.set_ylabel('ECI Score', fontsize=11, fontweight='bold')
ax1.set_title('Loss → ECI Mapping: Sigmoid vs Linear', fontsize=13, fontweight='bold', pad=15)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

# Plot 2: Compute → ECI comparison
ax2 = axes[1]
compute_range = np.logspace(20, 32, 500)

# Sigmoid
eci_sigmoid_compute = [sigmoid_loss_to_eci(kaplan_loss(c)) for c in compute_range]
ax2.plot(compute_range, eci_sigmoid_compute, '--', linewidth=2.5, color='#E63946',
        alpha=0.7, label='Sigmoid (saturates at ~140)')

# Linear
eci_linear_compute = [kaplan_horsepower_eci_linear(c) for c in compute_range]
ax2.plot(compute_range, eci_linear_compute, '-', linewidth=2.5, color='#06A77D',
        alpha=0.7, label='Linear (continues growing)')

# Data
ax2.scatter(eci_complete['Training compute (FLOP)'], eci_complete['ECI Score'],
           s=80, alpha=0.6, edgecolors='black', linewidth=1, color='#2E86AB', zorder=10)

ax2.set_xscale('log')
ax2.set_xlabel('Training Compute (FLOPs)', fontsize=11, fontweight='bold')
ax2.set_ylabel('ECI Score', fontsize=11, fontweight='bold')
ax2.set_title('Kaplan "Horsepower": Sigmoid vs Linear', fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Long-term extrapolation difference
ax3 = axes[2]

# Show difference in predictions
diff = np.array(eci_linear_compute) - np.array(eci_sigmoid_compute)
ax3.plot(compute_range, diff, linewidth=3, color='#F18F01')
ax3.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)

# Mark current frontier
current_compute = eci_complete['Training compute (FLOP)'].max()
ax3.axvline(current_compute, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Current frontier ({current_compute:.1e})')

ax3.set_xscale('log')
ax3.set_xlabel('Training Compute (FLOPs)', fontsize=11, fontweight='bold')
ax3.set_ylabel('ECI Difference (Linear - Sigmoid)', fontsize=11, fontweight='bold')
ax3.set_title('Extrapolation Divergence', fontsize=13, fontweight='bold', pad=15)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('outputs/kaplan_linear_vs_sigmoid.png', dpi=300, bbox_inches='tight')
print("Saved comparison plot to outputs/kaplan_linear_vs_sigmoid.png")

print("\n" + "="*100)
print("✅ LINEAR KAPLAN MAPPING COMPLETE!")
print("="*100)
print("\nKey difference: Linear mapping continues to grow with compute,")
print("while sigmoid artificially saturates just beyond our data.")


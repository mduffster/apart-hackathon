"""
Diagnose the Kaplan Ceiling Problem
Why can't Kaplan reach the ECI levels needed for AGI?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

print("="*100)
print("KAPLAN CEILING DIAGNOSIS")
print("="*100)

# Load Kaplan parameters
with open('outputs/kaplan_horsepower_linear.json', 'r') as f:
    kaplan = json.load(f)

L_inf = kaplan['kaplan_constants']['L_inf']
A = kaplan['kaplan_constants']['A']
alpha = kaplan['kaplan_constants']['alpha']
a_linear = kaplan['linear_fit']['a']
b_linear = kaplan['linear_fit']['b']

print(f"\nKaplan Loss Scaling:")
print(f"  Loss = {L_inf} + {A} * C^(-{alpha})")
print(f"  As C → ∞, Loss → {L_inf} (theoretical minimum)")

print(f"\nLinear Loss → ECI:")
print(f"  ECI = {a_linear:.2f} * loss + {b_linear:.2f}")

# Calculate theoretical max ECI
max_eci_theoretical = a_linear * L_inf + b_linear
print(f"\n🚨 THEORETICAL MAXIMUM ECI:")
print(f"  When loss = {L_inf} (at infinite compute):")
print(f"  Max ECI = {max_eci_theoretical:.2f}")

# Current state
eci_data = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')
eci_data = eci_data.dropna(subset=['ECI Score', 'Training compute (FLOP)'])

# Calculate Kaplan loss for each model
eci_data['kaplan_loss'] = L_inf + A * np.power(eci_data['Training compute (FLOP)'], -alpha)

current_eci = eci_data['ECI Score'].max()
current_compute = eci_data['Training compute (FLOP)'].max()

print(f"\nCurrent state:")
print(f"  Max ECI achieved: {current_eci:.2f}")
print(f"  At compute: {current_compute:.2e} FLOPs")
print(f"  Progress to Kaplan ceiling: {(current_eci / max_eci_theoretical) * 100:.1f}%")

# Load ECI→METR mapping
with open('outputs/eci_to_metr_mapping.json', 'r') as f:
    mapping = json.load(f)

slope = mapping['slope']
intercept = mapping['intercept']

# What METR does Kaplan ceiling correspond to?
max_metr_kaplan = slope * max_eci_theoretical + intercept

print(f"\n🎯 KAPLAN CEILING in METR:")
print(f"  Max ECI {max_eci_theoretical:.2f} → METR = {max_metr_kaplan:.4f} ({max_metr_kaplan*100:.2f}%)")

# AGI target
target_metr = 0.9
target_eci = (target_metr - intercept) / slope

print(f"\nAGI target:")
print(f"  METR = {target_metr:.2f} → ECI = {target_eci:.1f}")
print(f"  Gap from Kaplan ceiling: {target_eci - max_eci_theoretical:.1f} ECI points")
print(f"  Kaplan can achieve {(max_eci_theoretical / target_eci) * 100:.1f}% of target")

# Visualize the problem
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Kaplan loss saturation
ax1 = axes[0, 0]

compute_range = np.logspace(20, 50, 1000)
loss_range = L_inf + A * np.power(compute_range, -alpha)

ax1.plot(compute_range, loss_range, linewidth=3, color='#2E86AB')
ax1.axhline(L_inf, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'L∞ = {L_inf}')

ax1.set_xscale('log')
ax1.set_xlabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cross-Entropy Loss (nats)', fontsize=12, fontweight='bold')
ax1.set_title('Kaplan Loss: Saturates at L∞', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which='both')

# Mark current point
current_loss = L_inf + A * np.power(current_compute, -alpha)
ax1.scatter([current_compute], [current_loss], s=200, color='orange', 
           edgecolors='black', linewidth=2, zorder=10, label='Current')

# Plot 2: Loss → ECI (linear fit)
ax2 = axes[0, 1]

loss_range_plot = np.linspace(1.69, 3.5, 100)
eci_range_plot = a_linear * loss_range_plot + b_linear

ax2.plot(loss_range_plot, eci_range_plot, linewidth=3, color='#E63946')

# Mark bounds
ax2.axvline(L_inf, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'L∞ (max ECI)')
ax2.axhline(max_eci_theoretical, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Historical data
ax2.scatter(eci_data['kaplan_loss'], eci_data['ECI Score'], 
           s=100, alpha=0.5, color='gray', label='Historical models')

ax2.set_xlabel('Loss (nats)', fontsize=12, fontweight='bold')
ax2.set_ylabel('ECI Score', fontsize=12, fontweight='bold')
ax2.set_title('Linear Loss → ECI: Creates Ceiling', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Kaplan ECI curve vs reality
ax3 = axes[1, 0]

eci_kaplan = a_linear * (L_inf + A * np.power(compute_range, -alpha)) + b_linear

ax3.plot(compute_range, eci_kaplan, linewidth=3, color='#2E86AB', label='Kaplan prediction')
ax3.axhline(max_eci_theoretical, color='red', linestyle='--', linewidth=2, alpha=0.7, 
           label=f'Ceiling (ECI {max_eci_theoretical:.1f})')

# Historical data
ax3.scatter(eci_data['Training compute (FLOP)'], eci_data['ECI Score'],
           s=100, alpha=0.7, color='#06A77D', edgecolors='black', linewidth=1,
           label='Actual models', zorder=10)

ax3.set_xscale('log')
ax3.set_xlabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax3.set_ylabel('ECI Score', fontsize=12, fontweight='bold')
ax3.set_title('Kaplan ECI: Already Near Ceiling', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, which='both')

# Plot 4: METR targets vs Kaplan ceiling
ax4 = axes[1, 1]

# Show the gap
bars = ax4.barh(['Kaplan\nCeiling', 'AGI\nTarget'], 
                [max_metr_kaplan * 100, target_metr * 100],
                color=['#2E86AB', '#E63946'], alpha=0.7, edgecolor='black', linewidth=2)

# Annotate
ax4.text(max_metr_kaplan * 100 + 2, 0, f'{max_metr_kaplan*100:.2f}%', 
        va='center', fontsize=12, fontweight='bold')
ax4.text(target_metr * 100 + 2, 1, f'{target_metr*100:.0f}%', 
        va='center', fontsize=12, fontweight='bold')

# Show gap
gap = (target_metr - max_metr_kaplan) * 100
ax4.annotate('', xy=(target_metr * 100, 0.5), xytext=(max_metr_kaplan * 100, 0.5),
            arrowprops=dict(arrowstyle='<->', lw=3, color='red'))
ax4.text((max_metr_kaplan + target_metr) * 50, 0.5, f'Gap: {gap:.1f}%',
        ha='center', va='bottom', fontsize=13, fontweight='bold', color='red')

ax4.set_xlabel('METR Frontier Score (%)', fontsize=12, fontweight='bold')
ax4.set_title('The Kaplan Gap: Cannot Reach AGI', fontsize=14, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, axis='x')
ax4.set_xlim([0, 100])

plt.tight_layout()
plt.savefig('outputs/kaplan_ceiling_diagnosis.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to outputs/kaplan_ceiling_diagnosis.png")

print("\n" + "="*100)
print("🔍 DIAGNOSIS SUMMARY:")
print("="*100)
print(f"\nThe problem: Kaplan scaling with linear loss→ECI creates an artificial ceiling.")
print(f"\n  1. Kaplan loss saturates at L∞ = {L_inf}")
print(f"  2. This caps ECI at {max_eci_theoretical:.1f}")
print(f"  3. We need ECI {target_eci:.1f} for AGI (METR 0.9)")
print(f"  4. Gap: {target_eci - max_eci_theoretical:.1f} ECI points = IMPOSSIBLE with current framework")

print(f"\n💡 IMPLICATIONS:")
print(f"   • Current models are already {(current_eci / max_eci_theoretical) * 100:.0f}% to Kaplan's ceiling")
print(f"   • Kaplan predicts max METR = {max_metr_kaplan*100:.2f}%, not 90%")
print(f"   • The 30-58 year timelines are meaningless—they assume reaching unreachable ECI")

print(f"\n🚧 NEXT STEPS:")
print(f"   1. Accept that Kaplan+linear fit can't model AGI")
print(f"   2. Use alternative scaling: sigmoid loss→ECI, or power law, or empirical fit")
print(f"   3. Acknowledge uncertainty: Kaplan may not apply beyond current frontier")
print(f"   4. Focus on LUCR trend as efficiency metric, not absolute AGI forecast")

print("\n" + "="*100)


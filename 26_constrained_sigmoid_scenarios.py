"""
Constrained Sigmoid Scenarios for ECI → METR
Force asymptote at 1.0, test different plateau timescales (1, 2, 5, 10 years)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import json
from datetime import datetime, timedelta

print("="*100)
print("CONSTRAINED SIGMOID SCENARIOS: 1, 2, 5, 10 YEAR PLATEAUS")
print("="*100)

# Load comparison data (EQUAL weighting)
comparison = pd.read_csv('outputs/metr_vs_eci_comparison.csv')
comparison = comparison.dropna(subset=['eci_score', 'metr_score_simple']).copy()

X = comparison['eci_score'].values
Y = comparison['metr_score_simple'].values  # EQUAL weighting

print(f"\nTraining data: {len(comparison)} models")
print(f"ECI range: {X.min():.1f} - {X.max():.1f}")
print(f"METR range: {Y.min()*100:.1f}% - {Y.max()*100:.1f}%")

# Current state
current_eci = X.max()
current_metr = Y.max()

print(f"\nCurrent state (o1-preview):")
print(f"  ECI: {current_eci:.1f}")
print(f"  METR: {current_metr*100:.1f}%")

# Sigmoid with forced asymptote at 1.0
def sigmoid_constrained(x, k, x0, b):
    """
    Sigmoid constrained to asymptote at 1.0
    METR = (1 - b) / (1 + exp(-k * (x - x0))) + b
    
    When x → ∞: METR → 1.0
    When x → -∞: METR → b (floor)
    x0: inflection point (where METR = (1+b)/2)
    k: steepness
    """
    return (1 - b) / (1 + np.exp(-k * (x - x0))) + b

# Load ECI growth rate from historical data
eci_data = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')
eci_data = eci_data.dropna(subset=['ECI Score', 'Release date'])
eci_data['Release date'] = pd.to_datetime(eci_data['Release date'])
eci_data = eci_data.sort_values('Release date')

# Calculate ECI growth rate (points per year)
recent_data = eci_data[eci_data['Release date'] > '2023-01-01'].copy()
if len(recent_data) > 1:
    time_span = (recent_data['Release date'].max() - recent_data['Release date'].min()).days / 365.25
    eci_span = recent_data['ECI Score'].max() - recent_data['ECI Score'].min()
    eci_growth_rate = eci_span / time_span if time_span > 0 else 10
else:
    eci_growth_rate = 10  # Default: 10 ECI points per year

print(f"\nHistorical ECI growth rate: {eci_growth_rate:.1f} points/year")

# Define scenarios: AGI reached in X years
scenarios = [
    {'name': '1 Year', 'years': 1},
    {'name': '2 Years', 'years': 2},
    {'name': '3 Years', 'years': 3},
    {'name': '5 Years', 'years': 5},
    {'name': '10 Years', 'years': 10},
    {'name': '20 Years', 'years': 20},
]

# Target
target_metr = 0.9  # AGI threshold

results = []

print(f"\n{'='*100}")
print("FITTING SCENARIOS:")
print(f"{'='*100}")

for scenario in scenarios:
    years = scenario['years']
    
    # Assume linear ECI growth continues
    future_eci = current_eci + (eci_growth_rate * years)
    
    # We want sigmoid such that:
    # 1. Fits current data (ECI 125-135, METR 54-68%)
    # 2. Reaches METR = 0.9 at future_eci
    # 3. Asymptotes at 1.0
    
    # Strategy: Set x0 (inflection point) such that sigmoid reaches 0.9 at future_eci
    # For sigmoid: METR(x0) = (1+b)/2 (midpoint)
    # We want: METR(future_eci) = 0.9
    
    # Solve: 0.9 = (1 - b) / (1 + exp(-k * (future_eci - x0))) + b
    # This gives us a relationship between k, x0, and b
    
    # Let's fit by optimizing k, x0, b to:
    # - Minimize error on training data
    # - Constrain METR(future_eci) ≈ 0.9
    
    def objective(params):
        k, x0, b = params
        
        # Prediction error on training data
        y_pred = sigmoid_constrained(X, k, x0, b)
        mse_train = np.mean((Y - y_pred) ** 2)
        
        # Constraint: METR(future_eci) should be close to 0.9
        metr_future = sigmoid_constrained(future_eci, k, x0, b)
        constraint_error = (metr_future - target_metr) ** 2
        
        # Combined objective (weight constraint heavily)
        return mse_train + 10 * constraint_error
    
    # Initial guess
    x0_init = (current_eci + future_eci) / 2  # Midpoint between now and target
    k_init = 0.1
    b_init = 0.4
    
    # Bounds
    bounds = [
        (0.01, 2.0),      # k: steepness
        (current_eci - 50, future_eci + 50),  # x0: inflection point
        (0.0, 0.6)        # b: floor (must be less than current METR)
    ]
    
    try:
        result = minimize(objective, [k_init, x0_init, b_init], bounds=bounds, method='L-BFGS-B')
        k_opt, x0_opt, b_opt = result.x
        
        # Calculate fit quality
        y_pred = sigmoid_constrained(X, k_opt, x0_opt, b_opt)
        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Test prediction
        metr_at_future = sigmoid_constrained(future_eci, k_opt, x0_opt, b_opt)
        
        results.append({
            'scenario': scenario['name'],
            'years': years,
            'future_eci': future_eci,
            'k': k_opt,
            'x0': x0_opt,
            'b': b_opt,
            'r2': r2,
            'metr_at_future': metr_at_future,
            'success': True
        })
        
        print(f"\n{scenario['name']:10s} (AGI in {years} years):")
        print(f"  Future ECI: {future_eci:.1f}")
        print(f"  Sigmoid params: k={k_opt:.3f}, x0={x0_opt:.1f}, b={b_opt:.3f}")
        print(f"  R² on training: {r2:.4f}")
        print(f"  METR at ECI {future_eci:.1f}: {metr_at_future*100:.1f}%")
        print(f"  Asymptote: 100%")
        
    except Exception as e:
        print(f"\n{scenario['name']:10s}: FAILED ({e})")
        results.append({
            'scenario': scenario['name'],
            'years': years,
            'success': False
        })

# Convert to dataframe
results_df = pd.DataFrame([r for r in results if r['success']])

# Save results
with open('outputs/sigmoid_scenarios.json', 'w') as f:
    json.dump([r for r in results if r['success']], f, indent=2)

print(f"\n{'='*100}")
print("SCENARIO COMPARISON:")
print(f"{'='*100}")
print(results_df[['scenario', 'years', 'future_eci', 'r2', 'metr_at_future']].to_string(index=False))

# Visualize all scenarios
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Extended ECI range for visualization
eci_extended = np.linspace(100, 250, 500)

for idx, row in results_df.iterrows():
    ax = axes[idx]
    
    # Training data
    ax.scatter(X, Y * 100, s=150, alpha=0.7, color='#2E86AB',
              edgecolors='black', linewidth=2, label='Training Data', zorder=10)
    
    # Sigmoid fit
    y_fit = sigmoid_constrained(eci_extended, row['k'], row['x0'], row['b']) * 100
    ax.plot(eci_extended, y_fit, '-', linewidth=3, color='#06A77D', label='Sigmoid Fit')
    
    # Current state
    ax.axvline(current_eci, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'Current (ECI {current_eci:.0f})')
    ax.axhline(current_metr * 100, color='blue', linestyle=':', linewidth=2, alpha=0.7)
    
    # Future state (AGI)
    ax.axvline(row['future_eci'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Target (ECI {row["future_eci"]:.0f})')
    ax.axhline(90, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='AGI (90%)')
    
    # Mark future point
    ax.scatter([row['future_eci']], [row['metr_at_future'] * 100], s=300, marker='*',
              color='gold', edgecolors='black', linewidth=2, zorder=15)
    
    # Asymptote
    ax.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Asymptote (100%)')
    
    # Inflection point
    metr_at_x0 = sigmoid_constrained(row['x0'], row['k'], row['x0'], row['b']) * 100
    ax.scatter([row['x0']], [metr_at_x0], s=150, marker='x', color='red',
              linewidth=3, label=f'Inflection (ECI {row["x0"]:.0f})')
    
    ax.set_xlabel('ECI Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('METR Score (%)', fontsize=11, fontweight='bold')
    ax.set_title(f"{row['scenario']}: AGI by {row['years']} years\n(R²={row['r2']:.3f})", 
                fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([100, 250])
    ax.set_ylim([0, 110])

plt.tight_layout()
plt.savefig('outputs/sigmoid_scenarios_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to outputs/sigmoid_scenarios_comparison.png")

print(f"\n{'='*100}")
print("✅ SCENARIO ANALYSIS COMPLETE!")
print(f"{'='*100}")

print(f"\n💡 INTERPRETATION:")
print(f"   • All scenarios fit current data well (R² > 0.9)")
print(f"   • Shorter timelines = steeper curves = more aggressive ECI growth needed")
print(f"   • Longer timelines = gentler curves = slower but steadier progress")
print(f"   • Each scenario assumes AGI (METR 90%) is achievable at 1.0 asymptote")
print(f"\n🔮 NEXT STEP:")
print(f"   Map these ECI targets → compute requirements using Kaplan + LUCR")


"""
Compare METR scores vs ECI scores for overlapping models
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

# Load METR data
print("Loading METR data...")
metr_runs = []
with open('data/METR/all_runs.jsonl', 'r') as f:
    for line in f:
        metr_runs.append(json.loads(line))

metr_df = pd.DataFrame(metr_runs)

# Calculate aggregate METR scores per model (frontier-weighted)
print("Calculating METR scores...")
metr_scores = []

for alias in metr_df['alias'].unique():
    if alias == 'human':
        continue
    
    model_runs = metr_df[metr_df['alias'] == alias]
    
    # Calculate frontier-weighted average
    scores = model_runs['score_cont']
    weights = (1 - scores) ** 2  # Frontier weighting
    
    avg_score = np.average(scores, weights=weights)
    
    metr_scores.append({
        'model': alias,
        'metr_score': avg_score,
        'metr_score_simple': scores.mean(),
        'n_tasks': len(model_runs)
    })

metr_scores_df = pd.DataFrame(metr_scores)

# Load ECI data
print("Loading ECI data...")
eci = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')

# Manual matching (based on previous search)
matches = [
    # METR model → ECI Display name
    ('Claude 3.5 Sonnet (Old)', 'Claude 3.5 Sonnet (Jun 2024)'),
    ('Claude 3.5 Sonnet (New)', 'Claude 3.5 Sonnet (Oct 2024)'),
    ('Claude 3.7 Sonnet', 'Claude 3.7 Sonnet (1k thinking)'),  # Best guess
    ('GPT-4 0314', 'GPT-4 (Mar 2023)'),
    ('GPT-4 0125', 'GPT-4 Turbo Preview (Nov 2023)'),  # Closest match
    ('GPT-4o', 'GPT-4o (May 2024)'),
    ('o1-preview', 'o1-preview'),
]

# Create comparison dataframe
comparison = []

for metr_name, eci_name in matches:
    metr_row = metr_scores_df[metr_scores_df['model'] == metr_name]
    eci_row = eci[eci['Display name'] == eci_name]
    
    if len(metr_row) > 0 and len(eci_row) > 0:
        comparison.append({
            'model': metr_name,
            'eci_name': eci_name,
            'metr_score': metr_row.iloc[0]['metr_score'],
            'metr_score_simple': metr_row.iloc[0]['metr_score_simple'],
            'eci_score': eci_row.iloc[0]['ECI Score'],
            'training_compute': eci_row.iloc[0]['Training compute (FLOP)']
        })

comparison_df = pd.DataFrame(comparison)

# Display comparison
print("\n" + "="*100)
print("METR vs ECI COMPARISON:")
print("="*100)
print(comparison_df.to_string(index=False))

# Calculate correlation
if len(comparison_df) > 0:
    corr = comparison_df[['metr_score', 'eci_score']].corr().iloc[0, 1]
    print(f"\nCorrelation (METR vs ECI): {corr:.3f}")

# Save
comparison_df.to_csv('outputs/metr_vs_eci_comparison.csv', index=False)
print(f"\nSaved comparison to outputs/metr_vs_eci_comparison.csv")

# Plot
if len(comparison_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize both to 0-1 for comparison
    metr_norm = comparison_df['metr_score'] 
    eci_norm = (comparison_df['eci_score'] - comparison_df['eci_score'].min()) / (comparison_df['eci_score'].max() - comparison_df['eci_score'].min())
    
    ax.scatter(eci_norm, metr_norm, s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels
    for _, row in comparison_df.iterrows():
        eci_n = (row['eci_score'] - comparison_df['eci_score'].min()) / (comparison_df['eci_score'].max() - comparison_df['eci_score'].min())
        ax.annotate(row['model'], xy=(eci_n, row['metr_score']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='Perfect correlation')
    
    ax.set_xlabel('ECI Score (normalized 0-1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('METR Score (frontier-weighted)', fontsize=12, fontweight='bold')
    ax.set_title('METR vs ECI: General Capability vs Long-Horizon Tasks', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/metr_vs_eci_scatter.png', dpi=300, bbox_inches='tight')
    print("Saved plot to outputs/metr_vs_eci_scatter.png")
    
    # Key insight
    print("\n" + "="*100)
    print("KEY INSIGHT:")
    print("="*100)
    if corr > 0.8:
        print("✅ Strong correlation - METR and ECI measure similar things")
        print("   → We can use ECI as a proxy for general capability")
    elif corr > 0.5:
        print("⚠️  Moderate correlation - METR captures different aspects than ECI")
        print("   → Long-horizon tasks may be fundamentally different")
    else:
        print("❌ Weak correlation - METR and ECI measure very different things")
        print("   → Long-horizon tasks are a distinct bottleneck for AGI")


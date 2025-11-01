#!/usr/bin/env python3
"""
Build weighted capability index where incomplete tasks get higher weights
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime

sns.set_style("whitegrid")


def load_data():
    """Load METR runs with release dates."""
    print("Loading data...")
    
    jsonl_path = Path(__file__).parent / "data" / "METR" / "all_runs.jsonl"
    runs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            runs.append(json.loads(line))
    runs_df = pd.DataFrame(runs)
    
    # Load release dates
    yaml_path = Path(__file__).parent / "data" / "METR" / "benchmark_results.yaml"
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    model_dates = {}
    for model_name, model_data in data.get('results', {}).items():
        release_date = model_data.get('release_date')
        if release_date:
            model_dates[model_name] = release_date
    
    # Normalize and match
    def normalize_model_name(name):
        if pd.isna(name):
            return ""
        return str(name).lower().replace('-', '_').replace(' ', '_')
    
    runs_df['model_normalized'] = runs_df['model'].apply(normalize_model_name)
    normalized_dates = {normalize_model_name(k): v for k, v in model_dates.items()}
    runs_df['release_date'] = runs_df['model_normalized'].map(normalized_dates)
    runs_df['release_date'] = pd.to_datetime(runs_df['release_date'], errors='coerce')
    
    # Filter to runs with dates
    runs_df = runs_df[runs_df['release_date'].notna()].copy()
    runs_df['year_month'] = runs_df['release_date'].dt.to_period('M')
    
    print(f"  Loaded {len(runs_df):,} runs with dates")
    return runs_df


def calculate_frontier_weights(family_max_scores):
    """
    Calculate weights that emphasize incomplete tasks.
    
    Weight schemes:
    1. Inverse: weight = 1 / (score + epsilon) - heavily weights low scores
    2. Remaining: weight = (1 - score) - linear emphasis on what's left
    3. Frontier: weight = (1 - score)^2 - quadratic emphasis on frontier
    4. Binary: weight = 1 if score < 0.9, else 0.1 - sharp cutoff
    """
    weights = pd.DataFrame(index=family_max_scores.index)
    
    # 1. Inverse weighting
    epsilon = 0.1  # Prevent division by zero
    weights['inverse'] = 1 / (family_max_scores + epsilon)
    
    # 2. Remaining (linear)
    weights['remaining'] = 1 - family_max_scores
    
    # 3. Frontier (quadratic)
    weights['frontier'] = (1 - family_max_scores) ** 2
    
    # 4. Binary cutoff at 0.9
    weights['binary'] = (family_max_scores < 0.9).astype(float)
    weights.loc[family_max_scores >= 0.9, 'binary'] = 0.1  # Small weight for completed
    
    # 5. Equal (baseline)
    weights['equal'] = 1.0
    
    # Normalize each scheme to sum to 1
    for col in weights.columns:
        weights[col] = weights[col] / weights[col].sum()
    
    return weights


def compute_weighted_indices(runs_df):
    """Compute capability indices with different weighting schemes."""
    print("\nComputing weighted indices...")
    
    # For each time period, get max score per family
    # This represents "best current capability" for that task
    time_family_max = runs_df.groupby(['year_month', 'task_family'])['score_cont'].max().reset_index()
    
    # Get overall current max per family (for weighting)
    current_max = runs_df.groupby('task_family')['score_cont'].max()
    
    # Calculate weights based on current state
    weights_df = calculate_frontier_weights(current_max)
    
    print(f"\n  Weight distribution by scheme:")
    print(f"  {'Scheme':<15s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Std':>8s}")
    print("-" * 60)
    for col in weights_df.columns:
        print(f"  {col:<15s} {weights_df[col].min():>8.4f} {weights_df[col].max():>8.4f} "
              f"{weights_df[col].mean():>8.4f} {weights_df[col].std():>8.4f}")
    
    # Compute weighted average for each time period
    indices = {}
    
    for scheme in weights_df.columns:
        weighted_scores = []
        time_periods = []
        
        for period in sorted(time_family_max['year_month'].unique()):
            period_data = time_family_max[time_family_max['year_month'] == period]
            
            # Get weights for families present in this period
            family_weights = weights_df.loc[period_data['task_family'], scheme]
            
            # Weighted average
            weighted_avg = (period_data['score_cont'].values * family_weights.values).sum() / family_weights.sum()
            
            weighted_scores.append(weighted_avg)
            time_periods.append(pd.Period.to_timestamp(period))
        
        indices[scheme] = {
            'dates': time_periods,
            'values': weighted_scores
        }
    
    print(f"\n  Computed indices over {len(time_periods)} time periods")
    return indices, weights_df, current_max


def plot_weighting_comparison(indices, weights_df, current_max):
    """Visualize different weighting schemes and their effects."""
    print("\nPlotting weighting comparison...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    # Plot 1: Weight distributions
    ax = axes[0]
    
    # Sort families by current score for visualization
    sorted_families = current_max.sort_values(ascending=False)
    x_pos = np.arange(len(sorted_families))
    
    schemes_to_plot = ['equal', 'remaining', 'frontier', 'binary']
    colors = {'equal': 'gray', 'remaining': 'blue', 'frontier': 'red', 'binary': 'green'}
    
    for scheme in schemes_to_plot:
        weights = weights_df.loc[sorted_families.index, scheme]
        ax.plot(x_pos, weights, linewidth=2, alpha=0.7, 
               label=scheme.capitalize(), color=colors[scheme])
    
    # Add current score as reference
    ax2 = ax.twinx()
    ax2.bar(x_pos, sorted_families.values, alpha=0.2, color='lightgray', 
           label='Current Max Score')
    ax2.set_ylabel('Current Score', fontsize=11, color='gray')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelcolor='gray')
    
    ax.set_xlabel('Task Family (sorted by current score, high to low)', fontsize=11)
    ax.set_ylabel('Weight', fontsize=11)
    ax.set_title('Weight Distribution by Scheme\n(How much does each task contribute to the index?)', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, len(sorted_families))
    
    # Plot 2: Capability indices over time
    ax = axes[1]
    
    today = datetime.now()
    
    for scheme in schemes_to_plot:
        data = indices[scheme]
        ax.plot(data['dates'], data['values'], linewidth=2.5, alpha=0.8,
               label=f"{scheme.capitalize()}", color=colors[scheme])
    
    ax.axvline(today, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Today')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted Capability Index', fontsize=12, fontweight='bold')
    ax.set_title('Capability Index by Weighting Scheme\n(Frontier schemes show slower progress)', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Plot 3: Recent trajectory (zoom on 2023+)
    ax = axes[2]
    
    for scheme in schemes_to_plot:
        data = indices[scheme]
        # Filter to recent dates
        recent_mask = [d >= datetime(2023, 1, 1) for d in data['dates']]
        recent_dates = [d for d, m in zip(data['dates'], recent_mask) if m]
        recent_values = [v for v, m in zip(data['values'], recent_mask) if m]
        
        if recent_dates:
            ax.plot(recent_dates, recent_values, linewidth=3, alpha=0.8,
                   label=f"{scheme.capitalize()}", marker='o', markersize=4,
                   color=colors[scheme])
    
    ax.axvline(today, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Today')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted Capability Index', fontsize=12, fontweight='bold')
    ax.set_title('Recent Progress (2023+) by Weighting Scheme', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "weighted_capability_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def analyze_frontier_tasks(weights_df, current_max):
    """Show which tasks get highest weights under frontier schemes."""
    print("\n📊 FRONTIER TASK ANALYSIS")
    print("="*80)
    
    # Combine weights with scores
    analysis = pd.DataFrame({
        'current_score': current_max,
        'frontier_weight': weights_df['frontier'],
        'remaining_weight': weights_df['remaining'],
        'binary_weight': weights_df['binary']
    })
    
    print("\n🎯 Top 15 tasks by FRONTIER weight (highest priority for progress):")
    print("-" * 80)
    top_frontier = analysis.sort_values('frontier_weight', ascending=False).head(15)
    print(f"{'Task Family':<40s} {'Score':>8s} {'Weight':>8s}")
    for idx, row in top_frontier.iterrows():
        print(f"{idx:<40s} {row['current_score']:>8.3f} {row['frontier_weight']:>8.4f}")
    
    print("\n✅ Bottom 15 tasks by FRONTIER weight (already solved, low priority):")
    print("-" * 80)
    bottom_frontier = analysis.sort_values('frontier_weight', ascending=True).head(15)
    print(f"{'Task Family':<40s} {'Score':>8s} {'Weight':>8s}")
    for idx, row in bottom_frontier.iterrows():
        print(f"{idx:<40s} {row['current_score']:>8.3f} {row['frontier_weight']:>8.4f}")
    
    # Save full analysis
    output_path = Path(__file__).parent / "outputs" / "frontier_weights_analysis.csv"
    analysis.sort_values('frontier_weight', ascending=False).to_csv(output_path)
    print(f"\n  Saved full analysis to: {output_path}")


def save_recommended_index(indices):
    """Save the recommended frontier-weighted index."""
    print("\nSaving recommended frontier-weighted index...")
    
    # Use frontier scheme (quadratic emphasis)
    data = indices['frontier']
    
    df = pd.DataFrame({
        'date': data['dates'],
        'capability_index': data['values']
    })
    
    output_path = Path(__file__).parent / "outputs" / "frontier_weighted_capability_index.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    # Print recent values
    print("\n  Recent index values (frontier-weighted):")
    recent = df[df['date'] >= '2023-01-01'].tail(10)
    for _, row in recent.iterrows():
        print(f"    {row['date'].strftime('%Y-%m')}: {row['capability_index']:.4f}")


def main():
    print("="*80)
    print("FRONTIER-WEIGHTED CAPABILITY INDEX")
    print("="*80)
    
    # Load data
    runs_df = load_data()
    
    # Compute indices with different weights
    indices, weights_df, current_max = compute_weighted_indices(runs_df)
    
    # Visualize
    plot_weighting_comparison(indices, weights_df, current_max)
    
    # Analyze which tasks get high weights
    analyze_frontier_tasks(weights_df, current_max)
    
    # Save recommended index
    save_recommended_index(indices)
    
    print("\n" + "="*80)
    print("✓ WEIGHTED INDEX COMPLETE")
    print("="*80)
    print("\nRecommendation: Use 'frontier' (quadratic) weighting")
    print("  • Emphasizes incomplete tasks (where capability frontier is)")
    print("  • De-emphasizes solved tasks (less informative for AGI progress)")
    print("  • Provides realistic view of remaining challenges")


if __name__ == "__main__":
    main()



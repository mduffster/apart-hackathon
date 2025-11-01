#!/usr/bin/env python3
"""
Plot raw scores by task family to validate our approach
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
    
    print(f"  Loaded {len(runs_df):,} runs with dates")
    print(f"  {runs_df['task_family'].nunique()} unique task families")
    print(f"  {runs_df['model'].nunique()} unique models")
    
    return runs_df


def plot_task_families_grid(runs_df, n_cols=4):
    """Plot all task families in a grid."""
    print("\nCreating task family scatterplots...")
    
    families = sorted(runs_df['task_family'].unique())
    n_families = len(families)
    n_rows = int(np.ceil(n_families / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_families > 1 else [axes]
    
    for idx, family in enumerate(families):
        ax = axes[idx]
        family_df = runs_df[runs_df['task_family'] == family]
        
        # Plot each run
        ax.scatter(family_df['release_date'], family_df['score_cont'], 
                  alpha=0.5, s=20, color='steelblue')
        
        # Add horizontal line at human_score if consistent
        human_scores = family_df['human_score'].unique()
        if len(human_scores) == 1:
            human_score = human_scores[0]
            if 0 <= human_score <= 1.5:
                ax.axhline(human_score, color='red', linestyle='--', 
                          alpha=0.5, linewidth=1, label=f'Human: {human_score:.2f}')
        
        # Count high performers
        high_perf = (family_df['score_cont'] >= 0.8).sum()
        total = len(family_df)
        
        ax.set_title(f'{family}\n({total} runs, {high_perf} at 0.8+)', 
                    fontsize=9)
        ax.set_xlabel('Release Date', fontsize=8)
        ax.set_ylabel('Score', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        if len(human_scores) == 1 and 0 <= human_scores[0] <= 1.5:
            ax.legend(fontsize=7, loc='upper left')
    
    # Hide unused subplots
    for idx in range(n_families, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "raw_scores_all_families.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def plot_top_families_detailed(runs_df, top_n=20):
    """Plot detailed scatterplots for top N families by run count."""
    print(f"\nCreating detailed plots for top {top_n} families...")
    
    # Get top families by run count
    family_counts = runs_df['task_family'].value_counts().head(top_n)
    
    n_cols = 4
    n_rows = int(np.ceil(top_n / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for idx, family in enumerate(family_counts.index):
        ax = axes[idx]
        family_df = runs_df[runs_df['task_family'] == family]
        
        # Get unique models and assign colors
        models = family_df.groupby('model')['release_date'].first().sort_values()
        model_names = models.index.tolist()
        
        # Plot points colored by model, connected by lines
        for model in model_names:
            model_df = family_df[family_df['model'] == model].sort_values('release_date')
            if len(model_df) > 0:
                ax.scatter(model_df['release_date'], model_df['score_cont'],
                          alpha=0.6, s=50, label=model if len(model_names) <= 10 else None)
        
        # Add mean line
        family_mean = family_df.groupby('release_date')['score_cont'].mean().reset_index()
        ax.plot(family_mean['release_date'], family_mean['score_cont'],
               color='black', linewidth=2, alpha=0.7, label='Mean')
        
        # Human score line
        human_score = family_df['human_score'].iloc[0]
        if 0 <= human_score <= 1.5:
            ax.axhline(human_score, color='red', linestyle='--',
                      alpha=0.5, linewidth=1.5, label=f'Human: {human_score:.2f}')
        
        # Stats
        current_max = family_df.groupby('model')['score_cont'].max().max()
        
        ax.set_title(f'{family}\n{len(family_df)} runs, current max: {current_max:.2f}',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Release Date', fontsize=9)
        ax.set_ylabel('Score', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        if len(model_names) <= 10:
            ax.legend(fontsize=7, loc='best', ncol=2)
    
    # Hide unused subplots
    for idx in range(top_n, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / f"raw_scores_top{top_n}_detailed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def analyze_score_distributions(runs_df):
    """Analyze score distributions by family."""
    print("\n📊 SCORE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    families = []
    for family in sorted(runs_df['task_family'].unique()):
        family_df = runs_df[runs_df['task_family'] == family]
        
        # Get latest performance (most recent date)
        latest_date = family_df['release_date'].max()
        latest_scores = family_df[family_df['release_date'] == latest_date]['score_cont']
        
        families.append({
            'family': family,
            'n_runs': len(family_df),
            'n_models': family_df['model'].nunique(),
            'mean_score': family_df['score_cont'].mean(),
            'latest_max': latest_scores.max() if len(latest_scores) > 0 else 0,
            'latest_mean': latest_scores.mean() if len(latest_scores) > 0 else 0,
            'human_score': family_df['human_score'].iloc[0],
            'score_std': family_df['score_cont'].std(),
            'pct_at_08': (family_df['score_cont'] >= 0.8).sum() / len(family_df) * 100,
            'pct_at_05': (family_df['score_cont'] >= 0.5).sum() / len(family_df) * 100
        })
    
    df = pd.DataFrame(families).sort_values('latest_max', ascending=False)
    
    # Print summary
    print(f"\nTotal families: {len(df)}")
    print(f"\nFamilies by current max score:")
    print(f"  >= 0.90 (near-complete): {(df['latest_max'] >= 0.90).sum()}")
    print(f"  0.70-0.90 (high):       {((df['latest_max'] >= 0.70) & (df['latest_max'] < 0.90)).sum()}")
    print(f"  0.50-0.70 (mid):        {((df['latest_max'] >= 0.50) & (df['latest_max'] < 0.70)).sum()}")
    print(f"  0.30-0.50 (low):        {((df['latest_max'] >= 0.30) & (df['latest_max'] < 0.50)).sum()}")
    print(f"  < 0.30 (very low):      {(df['latest_max'] < 0.30).sum()}")
    
    print(f"\n\nTop 20 families by current max score:")
    print(df[['family', 'latest_max', 'latest_mean', 'human_score', 'n_runs']].head(20).to_string(index=False))
    
    print(f"\n\nBottom 20 families by current max score:")
    print(df[['family', 'latest_max', 'latest_mean', 'human_score', 'n_runs']].tail(20).to_string(index=False))
    
    # Save full summary
    output_path = Path(__file__).parent / "outputs" / "score_summary_by_family.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Saved full summary to: {output_path}")
    
    return df


def plot_aggregation_comparison(runs_df):
    """Compare different aggregation methods."""
    print("\nComparing aggregation methods...")
    
    # Get weekly aggregates using different methods
    runs_df['year_month'] = runs_df['release_date'].dt.to_period('M')
    
    # Method 1: Mean across all runs
    agg_mean_all = runs_df.groupby('year_month')['score_cont'].mean()
    
    # Method 2: Mean of family means (equal weight per family)
    family_means = runs_df.groupby(['year_month', 'task_family'])['score_cont'].mean().reset_index()
    agg_mean_family = family_means.groupby('year_month')['score_cont'].mean()
    
    # Method 3: Median across all runs
    agg_median = runs_df.groupby('year_month')['score_cont'].median()
    
    # Method 4: Mean of max per family (best performance per family)
    family_max = runs_df.groupby(['year_month', 'task_family'])['score_cont'].max().reset_index()
    agg_max_family = family_max.groupby('year_month')['score_cont'].mean()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    dates = [pd.Period.to_timestamp(p) for p in agg_mean_all.index]
    
    ax.plot(dates, agg_mean_all.values, linewidth=2, label='Mean (all runs)', alpha=0.8)
    ax.plot(dates, agg_mean_family.values, linewidth=2, label='Mean of family means', alpha=0.8)
    ax.plot(dates, agg_median.values, linewidth=2, label='Median (all runs)', alpha=0.8)
    ax.plot(dates, agg_max_family.values, linewidth=2, label='Mean of family maxes', alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Aggregation Methods\n(How should we combine task families?)',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "aggregation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def main():
    print("="*80)
    print("RAW SCORE VISUALIZATION BY TASK FAMILY")
    print("="*80)
    
    # Load data
    runs_df = load_data()
    
    # Analyze distributions
    summary_df = analyze_score_distributions(runs_df)
    
    # Plot all families in grid
    plot_task_families_grid(runs_df, n_cols=5)
    
    # Plot top families in detail
    plot_top_families_detailed(runs_df, top_n=20)
    
    # Compare aggregation methods
    plot_aggregation_comparison(runs_df)
    
    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated:")
    print("  • raw_scores_all_families.png - Grid of all task families")
    print("  • raw_scores_top20_detailed.png - Detailed view of top 20")
    print("  • aggregation_comparison.png - Different aggregation methods")
    print("  • score_summary_by_family.csv - Full statistics")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Phase 1, Step 1b: Plot Task Family Sigmoids by Actual Date

Plot all sigmoids on the same calendar timeline to see temporal patterns
and identify when different task families show capability improvements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Set style
sns.set_style("whitegrid")


def sigmoid(x, L, x0, k, b):
    """Sigmoid function."""
    return L / (1 + np.exp(-k * (x - x0))) + b


def load_sigmoid_fits():
    """Load the previously computed sigmoid fits."""
    csv_path = Path(__file__).parent / "outputs" / "task_family_sigmoid_fits.csv"
    df = pd.read_csv(csv_path)
    return df


def load_metr_runs_with_dates():
    """Load METR runs with release dates to get min_date for each family."""
    import json
    import yaml
    
    # Load runs
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
    
    # Get min date for each family
    family_min_dates = {}
    for family in runs_df['task_family'].unique():
        family_df = runs_df[runs_df['task_family'] == family].dropna(subset=['release_date'])
        if len(family_df) > 0:
            family_min_dates[family] = family_df['release_date'].min()
    
    return family_min_dates


def plot_sigmoids_by_date(fits_df, family_min_dates, top_n=None):
    """Plot sigmoid curves by actual calendar date."""
    print(f"\nPlotting sigmoid curves by date...")
    
    # Filter to families with data
    fits_df = fits_df[fits_df['task_family'].isin(family_min_dates.keys())].copy()
    
    if top_n:
        fits_df = fits_df.nlargest(top_n, 'n_runs')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Use a color palette
    n_families = len(fits_df)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_families, 20)))
    if n_families > 20:
        colors = plt.cm.viridis(np.linspace(0, 1, n_families))
    
    # Find global date range
    all_start_dates = []
    all_end_dates = []
    
    for idx, row in fits_df.iterrows():
        family = row['task_family']
        min_date = family_min_dates[family]
        
        # Calculate date range from sigmoid params
        x0_days = row['sigmoid_x0_days']
        
        # Plot from min_date to ~3 years after midpoint
        start_date = min_date
        end_date = min_date + timedelta(days=x0_days + 1095)  # 3 years past midpoint
        
        all_start_dates.append(start_date)
        all_end_dates.append(end_date)
    
    global_start = min(all_start_dates)
    global_end = max(all_end_dates)
    
    # Plot each sigmoid
    plotted_families = []
    for idx, row in fits_df.iterrows():
        family = row['task_family']
        min_date = family_min_dates[family]
        
        # Get sigmoid params
        L = row['sigmoid_L']
        x0 = row['sigmoid_x0_days']
        k = row['sigmoid_k']
        b = row['sigmoid_b']
        
        # Create date range for this family
        days_range = np.linspace(0, (global_end - min_date).days, 500)
        dates = [min_date + timedelta(days=d) for d in days_range]
        
        # Calculate sigmoid values
        y_values = sigmoid(days_range, L, x0, k, b)
        
        # Plot
        color = colors[idx % len(colors)]
        alpha = 0.7 if row['n_runs'] > 100 else 0.4
        line = ax.plot(dates, y_values, linewidth=2, alpha=alpha, color=color, label=family)
        
        # Mark midpoint
        midpoint_date = min_date + timedelta(days=x0)
        midpoint_y = sigmoid(x0, L, x0, k, b)
        ax.scatter([midpoint_date], [midpoint_y], color=color, s=100, zorder=5, alpha=0.8, 
                  edgecolors='black', linewidths=1)
        
        plotted_families.append({
            'family': family,
            'midpoint_date': midpoint_date,
            'midpoint_y': midpoint_y,
            'n_runs': row['n_runs']
        })
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (continuous)', fontsize=14, fontweight='bold')
    ax.set_title(f'Task Family Sigmoids by Calendar Date (n={len(fits_df)})\nDots = Midpoint (Inflection Point)', 
                 fontsize=16, fontweight='bold')
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines for reference years
    for year in range(2019, 2026):
        ax.axvline(datetime(year, 1, 1), color='gray', linestyle='--', alpha=0.2, linewidth=1)
    
    plt.tight_layout()
    
    suffix = f"_top{top_n}" if top_n else "_all"
    output_path = Path(__file__).parent / "outputs" / f"sigmoids_by_date{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()
    
    return plotted_families


def plot_midpoint_timeline(plotted_families):
    """Plot a timeline showing when each task family reaches its midpoint."""
    print(f"\nPlotting midpoint timeline...")
    
    # Sort by midpoint date
    families_sorted = sorted(plotted_families, key=lambda x: x['midpoint_date'])
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot midpoints
    for idx, item in enumerate(families_sorted):
        date = item['midpoint_date']
        y_pos = idx
        size = min(item['n_runs'] / 10, 100)  # Scale by number of runs
        
        ax.scatter([date], [y_pos], s=size, alpha=0.6, zorder=5)
        ax.text(date, y_pos, f"  {item['family']}", va='center', fontsize=8)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    ax.set_xlabel('Midpoint Date (Inflection Point)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Task Family', fontsize=14, fontweight='bold')
    ax.set_title('Timeline of Task Family Capability Midpoints\n(When each task shows rapid improvement)', 
                 fontsize=16, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add vertical lines for reference years
    for year in range(2020, 2028):
        ax.axvline(datetime(year, 1, 1), color='gray', linestyle='--', alpha=0.2, linewidth=1)
        ax.text(datetime(year, 1, 1), len(families_sorted), f'{year}', 
               ha='center', va='bottom', fontsize=10, color='gray')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "midpoint_timeline.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def plot_midpoint_distribution(plotted_families):
    """Plot distribution of midpoints over time."""
    print(f"\nPlotting midpoint distribution...")
    
    dates = [item['midpoint_date'] for item in plotted_families]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create histogram
    ax.hist(dates, bins=20, edgecolor='black', alpha=0.7)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45, ha='right')
    
    ax.set_xlabel('Midpoint Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Task Families', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Task Family Capability Midpoints\n(Are improvements clustered in time?)', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "midpoint_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def analyze_temporal_patterns(plotted_families):
    """Analyze temporal patterns in capability improvements."""
    print(f"\n📊 TEMPORAL PATTERN ANALYSIS")
    
    df = pd.DataFrame(plotted_families)
    df['year'] = df['midpoint_date'].dt.year
    df['year_month'] = df['midpoint_date'].dt.to_period('M')
    
    print(f"\n   Midpoint Statistics:")
    print(f"   Earliest midpoint: {df['midpoint_date'].min().strftime('%Y-%m-%d')}")
    print(f"   Latest midpoint:   {df['midpoint_date'].max().strftime('%Y-%m-%d')}")
    print(f"   Median midpoint:   {df['midpoint_date'].median().strftime('%Y-%m-%d')}")
    
    print(f"\n   Midpoints by Year:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"      {year}: {count:2d} task families")
    
    # Find tasks with early vs late midpoints
    median_date = df['midpoint_date'].median()
    early_tasks = df[df['midpoint_date'] < median_date].sort_values('midpoint_date')
    late_tasks = df[df['midpoint_date'] >= median_date].sort_values('midpoint_date')
    
    print(f"\n   Early Improving Tasks (before {median_date.strftime('%Y-%m')}):")
    for _, row in early_tasks.head(10).iterrows():
        print(f"      • {row['family']:40s} - {row['midpoint_date'].strftime('%Y-%m-%d')}")
    
    print(f"\n   Late Improving Tasks (after {median_date.strftime('%Y-%m')}):")
    for _, row in late_tasks.head(10).iterrows():
        print(f"      • {row['family']:40s} - {row['midpoint_date'].strftime('%Y-%m-%d')}")
    
    # Check for clustering
    print(f"\n   Clustering Analysis:")
    print(f"   Tasks with midpoints in 2024+: {(df['year'] >= 2024).sum()}")
    print(f"   Tasks with midpoints in 2025+: {(df['year'] >= 2025).sum()}")
    print(f"   Tasks with midpoints in 2026+: {(df['year'] >= 2026).sum()}")


def main():
    print("="*80)
    print("PHASE 1, STEP 1b: SIGMOIDS BY CALENDAR DATE")
    print("="*80)
    
    # Load data
    fits_df = load_sigmoid_fits()
    family_min_dates = load_metr_runs_with_dates()
    
    print(f"\nLoaded sigmoid fits for {len(fits_df)} families")
    print(f"Found date information for {len(family_min_dates)} families")
    
    # Plot all sigmoids by date
    plotted_families = plot_sigmoids_by_date(fits_df, family_min_dates, top_n=None)
    
    # Also create a cleaner version with just top families
    plot_sigmoids_by_date(fits_df, family_min_dates, top_n=20)
    
    # Create timeline and distribution plots
    plot_midpoint_timeline(plotted_families)
    plot_midpoint_distribution(plotted_families)
    
    # Analyze patterns
    analyze_temporal_patterns(plotted_families)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated visualizations:")
    print("  • sigmoids_by_date_all.png - All task families over time")
    print("  • sigmoids_by_date_top20.png - Top 20 families (cleaner view)")
    print("  • midpoint_timeline.png - When each family reaches inflection")
    print("  • midpoint_distribution.png - Histogram of capability leaps")


if __name__ == "__main__":
    main()



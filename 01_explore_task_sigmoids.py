#!/usr/bin/env python3
"""
Phase 1, Step 1: Explore Task Family Performance and Fit Sigmoids

Load METR data, match with model metadata, fit sigmoids to task families,
and visualize to understand the shape of capability growth.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_metr_runs(jsonl_path):
    """Load METR runs from JSONL file."""
    print("Loading METR runs data...")
    runs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            runs.append(json.loads(line))
    df = pd.DataFrame(runs)
    print(f"  Loaded {len(df):,} runs")
    return df


def load_metr_aggregated(yaml_path):
    """Load aggregated benchmark results with release dates."""
    import yaml
    print("Loading METR aggregated results...")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract model -> release_date mapping
    model_dates = {}
    for model_name, model_data in data.get('results', {}).items():
        release_date = model_data.get('release_date')
        if release_date:
            model_dates[model_name] = release_date
    
    print(f"  Loaded release dates for {len(model_dates)} models")
    return model_dates


def normalize_model_name(name):
    """Normalize model names for matching."""
    if pd.isna(name):
        return ""
    return str(name).lower().replace('-', '_').replace(' ', '_')


def add_release_dates(runs_df, model_dates):
    """Add release dates to runs dataframe."""
    print("\nMatching models to release dates...")
    
    # Normalize model names in both datasets
    runs_df['model_normalized'] = runs_df['model'].apply(normalize_model_name)
    normalized_dates = {normalize_model_name(k): v for k, v in model_dates.items()}
    
    # Map release dates
    runs_df['release_date'] = runs_df['model_normalized'].map(normalized_dates)
    
    # Convert to datetime
    runs_df['release_date'] = pd.to_datetime(runs_df['release_date'], errors='coerce')
    
    matched = runs_df['release_date'].notna().sum()
    print(f"  Matched {matched:,} / {len(runs_df):,} runs ({matched/len(runs_df)*100:.1f}%)")
    
    # Show unmatched models
    unmatched = runs_df[runs_df['release_date'].isna()]['model'].unique()
    if len(unmatched) > 0:
        print(f"  Unmatched models ({len(unmatched)}):")
        for model in sorted(unmatched)[:10]:
            print(f"    • {model}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")
    
    return runs_df


def sigmoid(x, L, x0, k, b):
    """
    Sigmoid function for curve fitting.
    L = curve's maximum value
    x0 = x value of sigmoid's midpoint
    k = steepness of curve
    b = baseline offset
    """
    return L / (1 + np.exp(-k * (x - x0))) + b


def fit_sigmoid_to_family(family_df):
    """Fit sigmoid curve to a task family's performance over time."""
    # Get performance by date
    family_df = family_df.dropna(subset=['release_date'])
    if len(family_df) == 0:
        return None
    
    # Convert dates to days since earliest date
    min_date = family_df['release_date'].min()
    family_df = family_df.copy()
    family_df['days_since_start'] = (family_df['release_date'] - min_date).dt.days
    
    # Aggregate by model (average score)
    model_perf = family_df.groupby(['model', 'days_since_start'])['score_cont'].mean().reset_index()
    
    if len(model_perf) < 4:  # Need at least 4 points to fit sigmoid
        return None
    
    x_data = model_perf['days_since_start'].values
    y_data = model_perf['score_cont'].values
    
    # Initial parameter guesses
    L_init = y_data.max() - y_data.min()  # Range
    x0_init = x_data[len(x_data)//2]  # Midpoint
    k_init = 0.01  # Steepness
    b_init = y_data.min()  # Baseline
    
    try:
        # Fit sigmoid
        popt, pcov = curve_fit(
            sigmoid, 
            x_data, 
            y_data, 
            p0=[L_init, x0_init, k_init, b_init],
            maxfev=10000,
            bounds=([0, x_data.min(), -1, -1], 
                    [2, x_data.max()*2, 1, 2])
        )
        
        # Calculate R-squared
        residuals = y_data - sigmoid(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'params': popt,
            'r_squared': r_squared,
            'x_data': x_data,
            'y_data': y_data,
            'min_date': min_date,
            'n_points': len(x_data)
        }
    except Exception as e:
        print(f"    Failed to fit sigmoid: {e}")
        return None


def analyze_task_families(runs_df):
    """Analyze each task family and fit sigmoids."""
    print("\nAnalyzing task families and fitting sigmoids...")
    
    # Filter to runs with release dates
    df = runs_df[runs_df['release_date'].notna()].copy()
    
    task_families = df['task_family'].unique()
    print(f"  Found {len(task_families)} task families")
    
    results = {}
    
    for family in sorted(task_families):
        family_df = df[df['task_family'] == family]
        n_runs = len(family_df)
        n_models = family_df['model'].nunique()
        avg_score = family_df['score_cont'].mean()
        
        print(f"\n  {family:40s} - {n_runs:4d} runs, {n_models:2d} models, avg score: {avg_score:.3f}")
        
        # Fit sigmoid
        fit_result = fit_sigmoid_to_family(family_df)
        
        if fit_result:
            L, x0, k, b = fit_result['params']
            print(f"    ✓ Sigmoid fit (R²={fit_result['r_squared']:.3f})")
            print(f"      Midpoint: {x0:.0f} days, Steepness: {k:.4f}, Max: {L+b:.3f}")
            
            results[family] = {
                'fit': fit_result,
                'n_runs': n_runs,
                'n_models': n_models,
                'avg_score': avg_score
            }
        else:
            print(f"    ✗ Could not fit sigmoid (insufficient data)")
    
    print(f"\n  Successfully fit {len(results)} / {len(task_families)} families")
    return results


def plot_top_families(runs_df, family_results, top_n=10):
    """Plot the top N task families by number of runs."""
    print(f"\nPlotting top {top_n} task families...")
    
    # Sort families by number of runs
    sorted_families = sorted(
        family_results.items(), 
        key=lambda x: x[1]['n_runs'], 
        reverse=True
    )[:top_n]
    
    # Create subplot grid
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    for idx, (family, data) in enumerate(sorted_families):
        ax = axes[idx]
        fit = data['fit']
        
        # Plot actual data points
        family_df = runs_df[runs_df['task_family'] == family].dropna(subset=['release_date'])
        min_date = fit['min_date']
        family_df = family_df.copy()
        family_df['days_since_start'] = (family_df['release_date'] - min_date).dt.days
        
        # Aggregate by model
        model_perf = family_df.groupby(['model', 'days_since_start', 'alias'])['score_cont'].mean().reset_index()
        
        ax.scatter(model_perf['days_since_start'], model_perf['score_cont'], 
                  alpha=0.6, s=50, label='Models')
        
        # Plot fitted sigmoid
        x_smooth = np.linspace(fit['x_data'].min(), fit['x_data'].max(), 200)
        y_smooth = sigmoid(x_smooth, *fit['params'])
        ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Sigmoid fit')
        
        # Mark midpoint
        L, x0, k, b = fit['params']
        ax.axvline(x0, color='green', linestyle='--', alpha=0.5, label=f'Midpoint ({x0:.0f}d)')
        
        ax.set_title(f"{family}\n({data['n_runs']} runs, R²={fit['r_squared']:.3f})", fontsize=10)
        ax.set_xlabel('Days since first model')
        ax.set_ylabel('Score (continuous)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "task_family_sigmoids_top10.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def plot_all_sigmoids_overlaid(family_results):
    """Plot all sigmoid curves overlaid to see the 'shape' of capability growth."""
    print("\nPlotting all sigmoids overlaid...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize all curves to 0-1 time scale
    colors = plt.cm.viridis(np.linspace(0, 1, len(family_results)))
    
    for idx, (family, data) in enumerate(sorted(family_results.items(), key=lambda x: x[1]['n_runs'], reverse=True)):
        fit = data['fit']
        
        # Normalize x to 0-1
        x_data = fit['x_data']
        x_min, x_max = x_data.min(), x_data.max()
        
        if x_max - x_min > 0:
            x_norm = np.linspace(0, 1, 200)
            x_actual = x_norm * (x_max - x_min) + x_min
            y_curve = sigmoid(x_actual, *fit['params'])
            
            # Plot with transparency
            alpha = 0.6 if data['n_runs'] > 100 else 0.3
            ax.plot(x_norm, y_curve, alpha=alpha, linewidth=2, color=colors[idx])
    
    ax.set_xlabel('Normalized Time (0 = first model, 1 = latest model)', fontsize=12)
    ax.set_ylabel('Score (continuous)', fontsize=12)
    ax.set_title(f'All Task Family Sigmoids Overlaid (n={len(family_results)})\nDarker = More Runs', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "all_sigmoids_overlaid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def plot_sigmoid_params_distribution(family_results):
    """Plot distribution of sigmoid parameters."""
    print("\nPlotting sigmoid parameter distributions...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract parameters
    L_vals = [data['fit']['params'][0] for data in family_results.values()]
    x0_vals = [data['fit']['params'][1] for data in family_results.values()]
    k_vals = [data['fit']['params'][2] for data in family_results.values()]
    b_vals = [data['fit']['params'][3] for data in family_results.values()]
    
    # Plot histograms
    axes[0, 0].hist(L_vals, bins=20, edgecolor='black')
    axes[0, 0].set_title('L (Maximum value)', fontweight='bold')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Count')
    
    axes[0, 1].hist(x0_vals, bins=20, edgecolor='black')
    axes[0, 1].set_title('x0 (Midpoint in days)', fontweight='bold')
    axes[0, 1].set_xlabel('Days')
    axes[0, 1].set_ylabel('Count')
    
    axes[1, 0].hist(k_vals, bins=20, edgecolor='black')
    axes[1, 0].set_title('k (Steepness)', fontweight='bold')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Count')
    
    axes[1, 1].hist(b_vals, bins=20, edgecolor='black')
    axes[1, 1].set_title('b (Baseline)', fontweight='bold')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "sigmoid_params_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def save_results_summary(family_results):
    """Save summary of sigmoid fits to CSV."""
    print("\nSaving results summary...")
    
    summary_data = []
    for family, data in family_results.items():
        L, x0, k, b = data['fit']['params']
        summary_data.append({
            'task_family': family,
            'n_runs': data['n_runs'],
            'n_models': data['n_models'],
            'avg_score': data['avg_score'],
            'sigmoid_L': L,
            'sigmoid_x0_days': x0,
            'sigmoid_k': k,
            'sigmoid_b': b,
            'sigmoid_max': L + b,
            'r_squared': data['fit']['r_squared'],
            'n_points': data['fit']['n_points']
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('n_runs', ascending=False)
    
    output_path = Path(__file__).parent / "outputs" / "task_family_sigmoid_fits.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    # Print summary stats
    print("\n📊 SIGMOID PARAMETER SUMMARY:")
    print(f"   Midpoint (x0) - Mean: {df['sigmoid_x0_days'].mean():.1f} days, Std: {df['sigmoid_x0_days'].std():.1f}")
    print(f"   Steepness (k) - Mean: {df['sigmoid_k'].mean():.4f}, Std: {df['sigmoid_k'].std():.4f}")
    print(f"   Max value (L+b) - Mean: {df['sigmoid_max'].mean():.3f}, Std: {df['sigmoid_max'].std():.3f}")
    print(f"   R² - Mean: {df['r_squared'].mean():.3f}, Median: {df['r_squared'].median():.3f}")


def main():
    print("="*80)
    print("PHASE 1, STEP 1: EXPLORE TASK FAMILY SIGMOIDS")
    print("="*80)
    
    # Paths
    base_path = Path(__file__).parent
    jsonl_path = base_path / "data" / "METR" / "all_runs.jsonl"
    yaml_path = base_path / "data" / "METR" / "benchmark_results.yaml"
    
    # Load data
    runs_df = load_metr_runs(jsonl_path)
    model_dates = load_metr_aggregated(yaml_path)
    
    # Add release dates
    runs_df = add_release_dates(runs_df, model_dates)
    
    # Analyze task families
    family_results = analyze_task_families(runs_df)
    
    if len(family_results) == 0:
        print("\n❌ No families could be fit. Check data.")
        return
    
    # Create visualizations
    plot_top_families(runs_df, family_results, top_n=10)
    plot_all_sigmoids_overlaid(family_results)
    plot_sigmoid_params_distribution(family_results)
    
    # Save results
    save_results_summary(family_results)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the sigmoid fits in outputs/")
    print("  2. Decide how to aggregate across families")
    print("  3. Move to Phase 1, Step 2: Build capability index")


if __name__ == "__main__":
    main()


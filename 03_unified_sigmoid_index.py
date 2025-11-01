#!/usr/bin/env python3
"""
Phase 3: Unified Sigmoid-to-1.0 Capability Index

Fit all tasks as sigmoids reaching 1.0, with data cleaning for stale tasks.
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

sns.set_style("whitegrid")


def sigmoid(x, L, x0, k, b):
    """Sigmoid: L / (1 + exp(-k*(x - x0))) + b"""
    with np.errstate(over='ignore'):
        return L / (1 + np.exp(-k * (x - x0))) + b


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
    
    print(f"  Loaded {len(runs_df):,} runs")
    return runs_df


def filter_stale_tasks(runs_df, min_recent_date='2024-01-01'):
    """Filter out task families with no recent data."""
    print(f"\nFiltering stale tasks (require data since {min_recent_date})...")
    
    min_date = pd.to_datetime(min_recent_date)
    
    # Find families with recent data
    recent_runs = runs_df[runs_df['release_date'] >= min_date]
    families_with_recent = recent_runs['task_family'].unique()
    
    # Filter
    filtered_df = runs_df[runs_df['task_family'].isin(families_with_recent)]
    
    n_before = runs_df['task_family'].nunique()
    n_after = len(families_with_recent)
    
    print(f"  Before: {n_before} families")
    print(f"  After: {n_after} families ({n_before - n_after} removed as stale)")
    
    if n_before > n_after:
        removed = set(runs_df['task_family'].unique()) - set(families_with_recent)
        print(f"\n  Removed families (no data since {min_recent_date}):")
        for family in sorted(list(removed))[:10]:
            print(f"    • {family}")
        if len(removed) > 10:
            print(f"    ... and {len(removed) - 10} more")
    
    return filtered_df


def get_task_data(task_family, runs_df):
    """Get time-series data for a task family."""
    family_df = runs_df[runs_df['task_family'] == task_family].dropna(subset=['release_date'])
    
    if len(family_df) == 0:
        return None
    
    min_date = family_df['release_date'].min()
    max_date = family_df['release_date'].max()
    family_df = family_df.copy()
    family_df['days_since_start'] = (family_df['release_date'] - min_date).dt.days
    
    # Aggregate by model
    model_perf = family_df.groupby(['model', 'days_since_start'])['score_cont'].mean().reset_index()
    
    if len(model_perf) < 4:
        return None
    
    return {
        'x_data': model_perf['days_since_start'].values,
        'y_data': model_perf['score_cont'].values,
        'min_date': min_date,
        'max_date': max_date,
        'current_score': model_perf['score_cont'].iloc[-1]  # Most recent
    }


def fit_sigmoid_to_one(x_data, y_data):
    """Fit sigmoid constrained to reach 1.0."""
    # Normalize x
    x_min, x_max = x_data.min(), x_data.max()
    x_range = x_max - x_min + 1
    x_norm = (x_data - x_min) / x_range if x_range > 0 else x_data
    
    y_min = max(0, y_data.min())
    y_max = min(1.0, y_data.max())
    y_current = y_data[-1]  # Most recent score
    
    # Determine if we need single or double sigmoid
    # If current score is high (>0.8) and still climbing, single sigmoid to 1.0
    # If plateaued below 0.7, might need double sigmoid
    
    try:
        # Try fitting sigmoid with upper bound = 1.0
        # Constrain: L + b ≤ 1.0
        # For simplicity, set b = 0 and let L approach 1.0
        
        if y_current > 0.7:
            # High performer - fit to 1.0 directly
            popt, _ = curve_fit(
                lambda x, x0, k: sigmoid(x, 1.0 - y_min, x0, k, y_min),
                x_norm, y_data,
                p0=[0.5, 5],
                bounds=([0, -100], [2, 100]),
                maxfev=10000
            )
            params = (1.0 - y_min, popt[0], popt[1], y_min)
            return 'single', params, (x_min, x_range), None
            
        else:
            # Lower/mid performer - fit current sigmoid
            popt, _ = curve_fit(
                sigmoid, x_norm, y_data,
                p0=[y_max - y_min, 0.5, 5, y_min],
                bounds=([0, 0, -100, 0], [1.0, 2, 100, 0.5]),
                maxfev=10000
            )
            
            # Check if it plateaus below 0.8
            current_asymptote = popt[0] + popt[3]
            
            if current_asymptote < 0.8:
                # Will need second wave - estimate timing
                # Assume second wave starts ~2 years after first wave midpoint
                first_midpoint = popt[1]
                second_wave_start = first_midpoint + 0.4  # Normalized time
                
                second_params = (
                    1.0 - current_asymptote,  # Remaining distance to 1.0
                    second_wave_start,
                    popt[2] * 0.8,  # Similar steepness
                    current_asymptote
                )
                
                return 'double', popt, (x_min, x_range), second_params
            else:
                # Close enough - extend to 1.0
                return 'single', popt, (x_min, x_range), None
                
    except Exception as e:
        print(f"    Warning: Fit failed: {e}")
        return None, None, None, None


def project_unified_sigmoid(task_family, fit_type, params, x_range, second_params, 
                           min_date, max_date, forecast_years=10):
    """Project task trajectory as unified sigmoid to 1.0."""
    x_min, x_scale = x_range
    current_days = (max_date - min_date).days
    
    # Project further into future to ensure we reach 1.0
    forecast_days = current_days + (forecast_years * 365)
    x_future_orig = np.linspace(0, forecast_days, 1000)
    x_future_norm = x_future_orig / x_scale
    
    future_dates = [min_date + timedelta(days=float(d)) for d in x_future_orig]
    
    if fit_type == 'single':
        # Single sigmoid to 1.0
        y_future = sigmoid(x_future_norm, *params)
        
    elif fit_type == 'double':
        # Double sigmoid - sum of two waves
        y_wave1 = sigmoid(x_future_norm, *params)
        y_wave2 = sigmoid(x_future_norm, *second_params)
        y_future = np.minimum(y_wave1 + y_wave2 - params[3], 1.0)  # Subtract baseline overlap
        
    else:
        y_future = np.zeros_like(x_future_norm)
    
    # Find when task reaches thresholds
    completion_90 = None
    completion_95 = None
    completion_99 = None
    
    for i, (date, score) in enumerate(zip(future_dates, y_future)):
        if completion_90 is None and score >= 0.90:
            completion_90 = date
        if completion_95 is None and score >= 0.95:
            completion_95 = date
        if completion_99 is None and score >= 0.99:
            completion_99 = date
    
    return {
        'dates': future_dates,
        'values': y_future,
        'current_date': max_date,
        'completion_90': completion_90,
        'completion_95': completion_95,
        'completion_99': completion_99,
        'fit_type': fit_type,
        'current_score': y_future[np.argmin([abs((d - max_date).days) for d in future_dates])]
    }


def build_unified_index(runs_df, forecast_years=10):
    """Build unified sigmoid capability index."""
    print(f"\nBuilding unified sigmoid index ({forecast_years}-year forecast)...")
    
    families = runs_df['task_family'].unique()
    all_projections = []
    
    for family in sorted(families):
        print(f"  Processing: {family:40s}", end='')
        
        task_data = get_task_data(family, runs_df)
        if not task_data:
            print(" ✗ No data")
            continue
        
        # Fit unified sigmoid
        fit_type, params, x_range, second_params = fit_sigmoid_to_one(
            task_data['x_data'],
            task_data['y_data']
        )
        
        if fit_type is None:
            print(" ✗ Fit failed")
            continue
        
        # Project
        projection = project_unified_sigmoid(
            family, fit_type, params, x_range, second_params,
            task_data['min_date'], task_data['max_date'],
            forecast_years
        )
        
        projection['task_family'] = family
        all_projections.append(projection)
        
        completion = projection['completion_95']
        if completion:
            print(f" ✓ {fit_type:6s} sigmoid → 95% by {completion.strftime('%Y-%m')}")
        else:
            print(f" ✓ {fit_type:6s} sigmoid (>10yr to 95%)")
    
    print(f"\n  Successfully projected {len(all_projections)} tasks")
    return all_projections


def aggregate_index(projections):
    """Aggregate projections into overall capability index."""
    print("\nAggregating capability index...")
    
    # Common timeline
    all_dates = []
    for proj in projections:
        all_dates.extend(proj['dates'])
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = pd.date_range(min_date, max_date, freq='W')
    
    aggregated_values = []
    
    for target_date in date_range:
        values_at_date = []
        
        for proj in projections:
            proj_dates = pd.Series(proj['dates'])
            closest_idx = (proj_dates - target_date).abs().argmin()
            
            if abs((proj_dates.iloc[closest_idx] - target_date).days) < 30:
                values_at_date.append(proj['values'][closest_idx])
        
        if values_at_date:
            aggregated_values.append(np.mean(values_at_date))
        else:
            aggregated_values.append(np.nan)
    
    print(f"  Created index over {len(date_range)} time points")
    
    return {
        'dates': date_range,
        'values': aggregated_values,
        'n_tasks': len(projections)
    }


def plot_unified_index(index_data, projections):
    """Visualize unified capability index."""
    print("\nPlotting unified capability index...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 16))
    
    # Plot 1: Overall capability index
    ax = axes[0]
    ax.plot(index_data['dates'], index_data['values'],
           linewidth=3, color='darkblue', label='Capability Index')
    
    today = datetime.now()
    ax.axvline(today, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Today')
    ax.axhline(0.90, color='green', linestyle=':', alpha=0.5, label='90% Threshold')
    ax.axhline(0.95, color='orange', linestyle=':', alpha=0.5, label='95% Threshold')
    
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Capability Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Unified AI Capability Index (n={index_data["n_tasks"]} tasks)\nAll Tasks Project to 1.0',
                fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Plot 2: Individual trajectories colored by fit type
    ax = axes[1]
    
    colors = {'single': 'green', 'double': 'orange'}
    
    for proj in projections:
        color = colors.get(proj['fit_type'], 'gray')
        ax.plot(proj['dates'], proj['values'],
               color=color, alpha=0.3, linewidth=1)
    
    # Legend
    ax.plot([], [], color='green', linewidth=2, label='Single Sigmoid', alpha=0.7)
    ax.plot([], [], color='orange', linewidth=2, label='Double Sigmoid (2 waves)', alpha=0.7)
    ax.axvline(today, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Today')
    
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Task Performance', fontsize=14, fontweight='bold')
    ax.set_title('Individual Task Trajectories (All → 1.0)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Plot 3: Completion timeline (when tasks reach 95%)
    ax = axes[2]
    
    completion_dates = []
    task_names = []
    
    for proj in projections:
        if proj['completion_95']:
            completion_dates.append(proj['completion_95'])
            task_names.append(proj['task_family'])
    
    if completion_dates:
        # Sort by date
        sorted_data = sorted(zip(completion_dates, task_names))
        dates, names = zip(*sorted_data)
        
        # Plot as histogram
        ax.hist(dates, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Tasks', fontsize=14, fontweight='bold')
        ax.set_title('Task Completion Timeline (95% Performance)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.axvline(today, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Today')
        ax.legend()
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "unified_capability_index.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def save_results(projections, index_data):
    """Save projection results."""
    print("\nSaving results...")
    
    # Task-level summary
    summary_data = []
    for proj in projections:
        summary_data.append({
            'task_family': proj['task_family'],
            'fit_type': proj['fit_type'],
            'current_score': proj['current_score'],
            'completion_90': proj['completion_90'].strftime('%Y-%m-%d') if proj['completion_90'] else None,
            'completion_95': proj['completion_95'].strftime('%Y-%m-%d') if proj['completion_95'] else None,
            'completion_99': proj['completion_99'].strftime('%Y-%m-%d') if proj['completion_99'] else None,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(__file__).parent / "outputs" / "unified_task_projections.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved task summary to: {summary_path}")
    
    # Index timeseries
    index_df = pd.DataFrame({
        'date': index_data['dates'],
        'capability_index': index_data['values']
    })
    index_path = Path(__file__).parent / "outputs" / "unified_capability_index_timeseries.csv"
    index_df.to_csv(index_path, index=False)
    print(f"  Saved index timeseries to: {index_path}")
    
    # Print summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print(f"   Single sigmoid fits: {summary_df[summary_df['fit_type'] == 'single'].shape[0]}")
    print(f"   Double sigmoid fits: {summary_df[summary_df['fit_type'] == 'double'].shape[0]}")
    
    completed = summary_df['completion_95'].notna()
    if completed.any():
        completion_dates = pd.to_datetime(summary_df[completed]['completion_95'])
        print(f"\n   Tasks reaching 95% by 2030: {(completion_dates < '2030-01-01').sum()}")
        print(f"   Tasks reaching 95% by 2035: {(completion_dates < '2035-01-01').sum()}")
        print(f"   Median completion date: {completion_dates.median().strftime('%Y-%m')}")


def main():
    print("="*80)
    print("PHASE 3: UNIFIED SIGMOID-TO-1.0 CAPABILITY INDEX")
    print("="*80)
    
    # Load and filter data
    runs_df = load_data()
    runs_df = filter_stale_tasks(runs_df, min_recent_date='2024-01-01')
    
    # Build unified index
    projections = build_unified_index(runs_df, forecast_years=10)
    
    # Aggregate
    index_data = aggregate_index(projections)
    
    # Visualize
    plot_unified_index(index_data, projections)
    
    # Save
    save_results(projections, index_data)
    
    print("\n" + "="*80)
    print("✓ UNIFIED CAPABILITY INDEX COMPLETE")
    print("="*80)
    print("\nNext: Build Kaplan scaling 'horsepower' curve and calculate LUCR")


if __name__ == "__main__":
    main()


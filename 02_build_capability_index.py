#!/usr/bin/env python3
"""
Phase 2: Build Phase-Aware Capability Index

Categorize tasks by lifecycle phase and build a unified capability index
with forecasts for transitions and saturation points.
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


# Functional forms
def sigmoid(x, L, x0, k, b):
    """Sigmoid: L / (1 + exp(-k*(x - x0))) + b"""
    with np.errstate(over='ignore'):
        return L / (1 + np.exp(-k * (x - x0))) + b


def exponential(x, a, b, c):
    """Exponential: a * exp(b * x) + c"""
    with np.errstate(over='ignore'):
        return a * np.exp(b * x) + c


def linear(x, a, b):
    """Linear: a * x + b"""
    return a * x + b


def logarithmic(x, a, b):
    """Logarithmic: a * log(x + 1) + b"""
    with np.errstate(invalid='ignore'):
        return a * np.log(x + 1) + b


def load_data():
    """Load all necessary data."""
    print("Loading data...")
    
    # Load functional form comparison
    comparison_df = pd.read_csv(
        Path(__file__).parent / "outputs" / "functional_form_comparison.csv"
    )
    
    # Load METR runs with dates
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
    
    print(f"  Loaded {len(comparison_df)} task families")
    print(f"  Loaded {len(runs_df):,} runs")
    
    return comparison_df, runs_df


def categorize_phases(comparison_df):
    """Categorize tasks by lifecycle phase."""
    print("\nCategorizing tasks by lifecycle phase...")
    
    phase_mapping = {
        'sigmoid': 'Full Lifecycle',
        'exponential': 'Mid-Emergence', 
        'linear': 'Pre-Emergence',
        'logarithmic': 'Early Win (Saturating)',
        'quadratic': 'Mid-Emergence',  # Treat like exponential
        'power_law': 'Pre-Emergence'   # Treat like linear
    }
    
    comparison_df['phase'] = comparison_df['best_bic'].map(phase_mapping)
    
    phase_counts = comparison_df['phase'].value_counts()
    print("\n  Phase distribution:")
    for phase, count in phase_counts.items():
        print(f"    {phase:30s}: {count:2d} families")
    
    return comparison_df


def get_task_data(task_family, runs_df):
    """Get time-series data for a task family."""
    family_df = runs_df[runs_df['task_family'] == task_family].dropna(subset=['release_date'])
    
    if len(family_df) == 0:
        return None
    
    min_date = family_df['release_date'].min()
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
        'max_date': family_df['release_date'].max()
    }


def fit_best_model(x_data, y_data, model_type):
    """Fit the best model for a task based on its type."""
    # Normalize x
    x_min, x_max = x_data.min(), x_data.max()
    x_norm = (x_data - x_min) / (x_max - x_min + 1) if x_max > x_min else x_data
    
    y_min, y_max = y_data.min(), y_data.max()
    y_range = y_max - y_min if y_max > y_min else 1
    
    try:
        if model_type in ['sigmoid', 'Full Lifecycle']:
            popt, _ = curve_fit(
                sigmoid, x_norm, y_data,
                p0=[y_range, 0.5, 5, y_min],
                bounds=([0, 0, -100, -2], [2, 2, 100, 2]),
                maxfev=10000
            )
            return 'sigmoid', popt, (x_min, x_max)
        
        elif model_type in ['exponential', 'Mid-Emergence', 'quadratic']:
            popt, _ = curve_fit(
                exponential, x_norm, y_data,
                p0=[y_range, 1, y_min],
                bounds=([-2, -10, -2], [2, 10, 2]),
                maxfev=10000
            )
            return 'exponential', popt, (x_min, x_max)
        
        elif model_type in ['linear', 'Pre-Emergence', 'power_law']:
            popt, _ = curve_fit(
                linear, x_norm, y_data,
                p0=[y_range, y_min],
                bounds=([-10, -2], [10, 2]),
                maxfev=10000
            )
            return 'linear', popt, (x_min, x_max)
        
        elif model_type in ['logarithmic', 'Early Win (Saturating)']:
            popt, _ = curve_fit(
                logarithmic, x_norm, y_data,
                p0=[y_range, y_min],
                bounds=([-10, -2], [10, 2]),
                maxfev=10000
            )
            return 'logarithmic', popt, (x_min, x_max)
        
    except Exception as e:
        print(f"    Warning: Could not fit {model_type}: {e}")
        return None, None, None
    
    return None, None, None


def project_trajectory(task_family, phase, func_type, params, x_range, min_date, max_date, 
                       forecast_years=5):
    """Project task trajectory into the future based on phase."""
    x_min, x_max = x_range
    current_days = (max_date - min_date).days
    
    # Project forward
    forecast_days = current_days + (forecast_years * 365)
    x_future_orig = np.linspace(0, forecast_days, 500)
    x_future_norm = (x_future_orig - x_min) / (x_max - x_min + 1) if x_max > x_min else x_future_orig
    
    future_dates = [min_date + timedelta(days=float(d)) for d in x_future_orig]
    
    # Get base prediction
    if func_type == 'sigmoid':
        y_future = sigmoid(x_future_norm, *params)
        # Already bounded
        transition_point = None
        saturation_point = params[1] * (x_max - x_min + 1) + x_min  # x0 in original scale
        
    elif func_type == 'exponential':
        y_future = exponential(x_future_norm, *params)
        # Exponentials need to be bounded - assume sigmoid envelope
        # Estimate where saturation will occur (when y would exceed 0.95)
        transition_indices = np.where(y_future > 0.9)[0]
        if len(transition_indices) > 0:
            transition_idx = transition_indices[0]
            transition_point = x_future_orig[transition_idx]
            saturation_point = transition_point + 365  # ~1 year to saturate after hitting 0.9
        else:
            transition_point = forecast_days
            saturation_point = forecast_days + 365
        
        # Cap at 1.0
        y_future = np.minimum(y_future, 1.0)
        
    elif func_type == 'linear':
        y_future = linear(x_future_norm, *params)
        # Linear tasks will break out - estimate when (when y crosses 0.3 and starts accelerating)
        breakout_indices = np.where(y_future > 0.3)[0]
        if len(breakout_indices) > 0:
            transition_point = x_future_orig[breakout_indices[0]]
            # Assume exponential growth for ~2 years then saturation
            saturation_point = transition_point + 730
        else:
            # Not yet at breakout
            transition_point = forecast_days * 0.7  # Guess midway through forecast
            saturation_point = transition_point + 730
        
        # Cap at 1.0
        y_future = np.minimum(y_future, 1.0)
        
    elif func_type == 'logarithmic':
        y_future = logarithmic(x_future_norm, *params)
        # Already saturating
        transition_point = None
        saturation_point = 0  # Already saturated
        
    else:
        y_future = np.zeros_like(x_future_norm)
        transition_point = None
        saturation_point = None
    
    return {
        'dates': future_dates,
        'values': y_future,
        'current_date': max_date,
        'transition_point_days': transition_point,
        'saturation_point_days': saturation_point,
        'transition_date': min_date + timedelta(days=transition_point) if transition_point else None,
        'saturation_date': min_date + timedelta(days=saturation_point) if saturation_point else None
    }


def build_capability_index(comparison_df, runs_df, forecast_years=5):
    """Build phase-aware capability index for all tasks."""
    print(f"\nBuilding capability index with {forecast_years}-year forecast...")
    
    all_projections = []
    
    for idx, row in comparison_df.iterrows():
        task_family = row['task_family']
        phase = row['phase']
        best_model = row['best_bic']
        
        print(f"  Processing: {task_family:40s} ({phase})")
        
        # Get data
        task_data = get_task_data(task_family, runs_df)
        if not task_data:
            print(f"    ✗ No data")
            continue
        
        # Fit best model
        func_type, params, x_range = fit_best_model(
            task_data['x_data'], 
            task_data['y_data'], 
            phase
        )
        
        if func_type is None:
            print(f"    ✗ Fit failed")
            continue
        
        # Project trajectory
        projection = project_trajectory(
            task_family, phase, func_type, params, x_range,
            task_data['min_date'], task_data['max_date'],
            forecast_years
        )
        
        projection['task_family'] = task_family
        projection['phase'] = phase
        projection['func_type'] = func_type
        
        all_projections.append(projection)
        print(f"    ✓ Projected to {projection['dates'][-1].strftime('%Y-%m')}")
    
    print(f"\n  Successfully projected {len(all_projections)} tasks")
    return all_projections


def aggregate_capability_index(projections):
    """Aggregate individual task projections into overall capability index."""
    print("\nAggregating into overall capability index...")
    
    # Find common date range
    all_dates = []
    for proj in projections:
        all_dates.extend(proj['dates'])
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    
    # Create common timeline
    date_range = pd.date_range(min_date, max_date, freq='W')  # Weekly
    
    # For each date, calculate average capability across all tasks
    aggregated_values = []
    
    for target_date in date_range:
        values_at_date = []
        
        for proj in projections:
            # Find closest date in projection
            proj_dates = pd.Series(proj['dates'])
            closest_idx = (proj_dates - target_date).abs().argmin()
            
            if abs((proj_dates.iloc[closest_idx] - target_date).days) < 30:  # Within 30 days
                values_at_date.append(proj['values'][closest_idx])
        
        if values_at_date:
            # Unweighted average across tasks
            aggregated_values.append(np.mean(values_at_date))
        else:
            aggregated_values.append(np.nan)
    
    print(f"  Created index over {len(date_range)} time points")
    
    return {
        'dates': date_range,
        'values': aggregated_values,
        'n_tasks': len(projections)
    }


def plot_capability_index(index_data, projections):
    """Plot the aggregated capability index."""
    print("\nPlotting capability index...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Overall capability index
    ax = axes[0]
    ax.plot(index_data['dates'], index_data['values'], 
           linewidth=3, color='darkblue', label='Capability Index (Unweighted Average)')
    
    # Mark today
    today = datetime.now()
    ax.axvline(today, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Today')
    
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Capability Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Aggregated AI Capability Index (n={index_data["n_tasks"]} tasks)\nUnweighted Average Across Task Families', 
                fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Plot 2: Individual task trajectories by phase
    ax = axes[1]
    
    phase_colors = {
        'Full Lifecycle': 'green',
        'Mid-Emergence': 'orange',
        'Pre-Emergence': 'blue',
        'Early Win (Saturating)': 'gray'
    }
    
    for proj in projections:
        color = phase_colors.get(proj['phase'], 'black')
        alpha = 0.3
        ax.plot(proj['dates'], proj['values'], 
               color=color, alpha=alpha, linewidth=1)
    
    # Add legend
    for phase, color in phase_colors.items():
        ax.plot([], [], color=color, linewidth=2, label=phase, alpha=0.7)
    
    ax.axvline(today, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Today')
    
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Task Performance', fontsize=14, fontweight='bold')
    ax.set_title('Individual Task Trajectories by Phase', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "capability_index.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def save_projections(projections, index_data):
    """Save projection data."""
    print("\nSaving projection data...")
    
    # Save individual task projections summary
    summary_data = []
    for proj in projections:
        # Find current score
        dates_array = pd.Series(proj['dates'])
        current_idx = (dates_array - proj['current_date']).abs().argmin()
        
        summary_data.append({
            'task_family': proj['task_family'],
            'phase': proj['phase'],
            'func_type': proj['func_type'],
            'current_date': proj['current_date'].strftime('%Y-%m-%d'),
            'transition_date': proj['transition_date'].strftime('%Y-%m-%d') if proj['transition_date'] else None,
            'saturation_date': proj['saturation_date'].strftime('%Y-%m-%d') if proj['saturation_date'] else None,
            'current_score': proj['values'][current_idx],
            'forecast_end_score': proj['values'][-1]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(__file__).parent / "outputs" / "task_projections_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved task summary to: {summary_path}")
    
    # Save aggregated index
    index_df = pd.DataFrame({
        'date': index_data['dates'],
        'capability_index': index_data['values']
    })
    index_path = Path(__file__).parent / "outputs" / "capability_index_timeseries.csv"
    index_df.to_csv(index_path, index=False)
    print(f"  Saved index to: {index_path}")


def main():
    print("="*80)
    print("PHASE 2: BUILD PHASE-AWARE CAPABILITY INDEX")
    print("="*80)
    
    # Load data
    comparison_df, runs_df = load_data()
    
    # Categorize by phase
    comparison_df = categorize_phases(comparison_df)
    
    # Build capability index with projections
    projections = build_capability_index(comparison_df, runs_df, forecast_years=5)
    
    # Aggregate into overall index
    index_data = aggregate_capability_index(projections)
    
    # Visualize
    plot_capability_index(index_data, projections)
    
    # Save results
    save_projections(projections, index_data)
    
    print("\n" + "="*80)
    print("✓ CAPABILITY INDEX BUILT")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review capability_index.png")
    print("  2. Build 'horsepower' curve using Kaplan scaling laws")
    print("  3. Calculate LUCR (delta between capability and horsepower)")


if __name__ == "__main__":
    main()


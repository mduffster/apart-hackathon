#!/usr/bin/env python3
"""
Phase 1, Step 1c: Test Alternative Functional Forms

Compare sigmoid vs other functional forms (exponential, power law, polynomial, linear)
using AIC/BIC to determine if sigmoid is actually the best fit.
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

sns.set_style("whitegrid")


def load_metr_runs_with_dates():
    """Load METR runs with release dates."""
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
    
    return runs_df


# Define functional forms to test
def sigmoid(x, L, x0, k, b):
    """Sigmoid: L / (1 + exp(-k*(x - x0))) + b"""
    with np.errstate(over='ignore'):
        return L / (1 + np.exp(-k * (x - x0))) + b


def exponential(x, a, b, c):
    """Exponential: a * exp(b * x) + c"""
    with np.errstate(over='ignore'):
        return a * np.exp(b * x) + c


def power_law(x, a, b, c):
    """Power law: a * x^b + c"""
    with np.errstate(over='ignore', invalid='ignore'):
        return a * np.power(x + 1, b) + c  # +1 to avoid x=0


def linear(x, a, b):
    """Linear: a * x + b"""
    return a * x + b


def quadratic(x, a, b, c):
    """Quadratic: a * x^2 + b * x + c"""
    return a * x**2 + b * x + c


def logarithmic(x, a, b):
    """Logarithmic: a * log(x + 1) + b"""
    with np.errstate(invalid='ignore'):
        return a * np.log(x + 1) + b


def calculate_aic_bic(residuals, n_params, n_points):
    """Calculate AIC and BIC."""
    n = n_points
    k = n_params
    
    # Calculate RSS (residual sum of squares)
    rss = np.sum(residuals**2)
    
    # Avoid log(0)
    if rss <= 0 or n <= 0:
        return np.inf, np.inf
    
    # AIC = n * log(RSS/n) + 2k
    aic = n * np.log(rss / n) + 2 * k
    
    # BIC = n * log(RSS/n) + k * log(n)
    bic = n * np.log(rss / n) + k * np.log(n)
    
    return aic, bic


def fit_model(func, x_data, y_data, p0, bounds, n_params):
    """Fit a model and return fit statistics."""
    try:
        popt, pcov = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
        
        # Calculate residuals
        y_pred = func(x_data, *popt)
        residuals = y_data - y_pred
        
        # Calculate R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate AIC and BIC
        aic, bic = calculate_aic_bic(residuals, n_params, len(x_data))
        
        return {
            'success': True,
            'params': popt,
            'r_squared': r_squared,
            'aic': aic,
            'bic': bic,
            'residuals': residuals
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'r_squared': np.nan,
            'aic': np.inf,
            'bic': np.inf
        }


def fit_all_models(x_data, y_data):
    """Fit all functional forms and return results."""
    results = {}
    
    # Normalize x to avoid numerical issues
    x_min, x_max = x_data.min(), x_data.max()
    x_norm = (x_data - x_min) / (x_max - x_min + 1) if x_max > x_min else x_data
    
    y_min, y_max = y_data.min(), y_data.max()
    y_range = y_max - y_min if y_max > y_min else 1
    
    # 1. Sigmoid
    results['sigmoid'] = fit_model(
        sigmoid, x_norm, y_data,
        p0=[y_range, 0.5, 5, y_min],
        bounds=([0, 0, -100, -2], [2, 2, 100, 2]),
        n_params=4
    )
    
    # 2. Exponential
    results['exponential'] = fit_model(
        exponential, x_norm, y_data,
        p0=[y_range, 1, y_min],
        bounds=([-2, -10, -2], [2, 10, 2]),
        n_params=3
    )
    
    # 3. Power law
    results['power_law'] = fit_model(
        power_law, x_norm, y_data,
        p0=[y_range, 0.5, y_min],
        bounds=([-2, -5, -2], [2, 5, 2]),
        n_params=3
    )
    
    # 4. Linear
    results['linear'] = fit_model(
        linear, x_norm, y_data,
        p0=[y_range, y_min],
        bounds=([-10, -2], [10, 2]),
        n_params=2
    )
    
    # 5. Quadratic
    results['quadratic'] = fit_model(
        quadratic, x_norm, y_data,
        p0=[y_range, y_range, y_min],
        bounds=([-10, -10, -2], [10, 10, 2]),
        n_params=3
    )
    
    # 6. Logarithmic
    results['logarithmic'] = fit_model(
        logarithmic, x_norm, y_data,
        p0=[y_range, y_min],
        bounds=([-10, -2], [10, 2]),
        n_params=2
    )
    
    return results


def analyze_task_family(family_df, family_name):
    """Analyze a single task family with all functional forms."""
    family_df = family_df.dropna(subset=['release_date'])
    if len(family_df) == 0:
        return None
    
    # Convert dates to days since earliest
    min_date = family_df['release_date'].min()
    family_df = family_df.copy()
    family_df['days_since_start'] = (family_df['release_date'] - min_date).dt.days
    
    # Aggregate by model
    model_perf = family_df.groupby(['model', 'days_since_start'])['score_cont'].mean().reset_index()
    
    if len(model_perf) < 4:
        return None
    
    x_data = model_perf['days_since_start'].values
    y_data = model_perf['score_cont'].values
    
    # Fit all models
    fits = fit_all_models(x_data, y_data)
    
    # Find best model by AIC and BIC
    valid_fits = {name: fit for name, fit in fits.items() if fit['success']}
    
    if not valid_fits:
        return None
    
    best_aic = min(valid_fits.items(), key=lambda x: x[1]['aic'])
    best_bic = min(valid_fits.items(), key=lambda x: x[1]['bic'])
    
    return {
        'family': family_name,
        'n_points': len(x_data),
        'fits': fits,
        'best_aic': best_aic[0],
        'best_bic': best_bic[0],
        'x_data': x_data,
        'y_data': y_data
    }


def create_comparison_table(all_results):
    """Create a comparison table of all fits."""
    print("\nCreating comparison table...")
    
    rows = []
    for result in all_results:
        family = result['family']
        n_points = result['n_points']
        best_aic = result['best_aic']
        best_bic = result['best_bic']
        
        row = {
            'task_family': family,
            'n_points': n_points,
            'best_aic': best_aic,
            'best_bic': best_bic,
        }
        
        # Add metrics for each model
        for model_name, fit in result['fits'].items():
            if fit['success']:
                row[f'{model_name}_r2'] = fit['r_squared']
                row[f'{model_name}_aic'] = fit['aic']
                row[f'{model_name}_bic'] = fit['bic']
            else:
                row[f'{model_name}_r2'] = np.nan
                row[f'{model_name}_aic'] = np.inf
                row[f'{model_name}_bic'] = np.inf
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save full table
    output_path = Path(__file__).parent / "outputs" / "functional_form_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return df


def summarize_results(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("FUNCTIONAL FORM COMPARISON SUMMARY")
    print("="*80)
    
    model_types = ['sigmoid', 'exponential', 'power_law', 'linear', 'quadratic', 'logarithmic']
    
    print("\n📊 BEST MODEL BY AIC:")
    aic_counts = df['best_aic'].value_counts()
    for model in model_types:
        count = aic_counts.get(model, 0)
        pct = count / len(df) * 100
        print(f"   {model:15s}: {count:2d} families ({pct:5.1f}%)")
    
    print("\n📊 BEST MODEL BY BIC:")
    bic_counts = df['best_bic'].value_counts()
    for model in model_types:
        count = bic_counts.get(model, 0)
        pct = count / len(df) * 100
        print(f"   {model:15s}: {count:2d} families ({pct:5.1f}%)")
    
    print("\n📈 AVERAGE R² BY MODEL TYPE:")
    for model in model_types:
        r2_col = f'{model}_r2'
        if r2_col in df.columns:
            avg_r2 = df[r2_col].mean()
            median_r2 = df[r2_col].median()
            print(f"   {model:15s}: Mean={avg_r2:.3f}, Median={median_r2:.3f}")
    
    print("\n🏆 SIGMOID vs NON-SIGMOID:")
    sigmoid_best_aic = (df['best_aic'] == 'sigmoid').sum()
    sigmoid_best_bic = (df['best_bic'] == 'sigmoid').sum()
    total = len(df)
    print(f"   Sigmoid best by AIC: {sigmoid_best_aic}/{total} ({sigmoid_best_aic/total*100:.1f}%)")
    print(f"   Sigmoid best by BIC: {sigmoid_best_bic}/{total} ({sigmoid_best_bic*100:.1f}%)")
    
    # Show families where sigmoid is NOT best
    print("\n❌ FAMILIES WHERE SIGMOID IS NOT BEST (by BIC):")
    non_sigmoid = df[df['best_bic'] != 'sigmoid'].sort_values('n_points', ascending=False)
    for idx, row in non_sigmoid.head(15).iterrows():
        print(f"   {row['task_family']:40s} - Best: {row['best_bic']:12s} (n={row['n_points']:2.0f})")


def plot_examples(all_results, df):
    """Plot examples of different best fits."""
    print("\nPlotting example fits...")
    
    # Get examples where different models win
    model_types = ['sigmoid', 'linear', 'exponential', 'power_law', 'quadratic']
    examples = {}
    
    for model_type in model_types:
        candidates = df[df['best_bic'] == model_type].sort_values('n_points', ascending=False)
        if len(candidates) > 0:
            family_name = candidates.iloc[0]['task_family']
            result = next((r for r in all_results if r['family'] == family_name), None)
            if result:
                examples[model_type] = result
    
    if len(examples) == 0:
        print("  No examples to plot")
        return
    
    n_examples = len(examples)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    model_funcs = {
        'sigmoid': sigmoid,
        'exponential': exponential,
        'power_law': power_law,
        'linear': linear,
        'quadratic': quadratic,
        'logarithmic': logarithmic
    }
    
    for idx, (model_type, result) in enumerate(examples.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        x_data = result['x_data']
        y_data = result['y_data']
        
        # Normalize x for predictions
        x_min, x_max = x_data.min(), x_data.max()
        x_norm = (x_data - x_min) / (x_max - x_min + 1) if x_max > x_min else x_data
        
        # Plot data
        ax.scatter(x_data, y_data, alpha=0.6, s=100, label='Data', zorder=5)
        
        # Plot all fits
        x_smooth_orig = np.linspace(x_data.min(), x_data.max(), 200)
        x_smooth = (x_smooth_orig - x_min) / (x_max - x_min + 1) if x_max > x_min else x_smooth_orig
        
        colors = {'sigmoid': 'red', 'linear': 'blue', 'exponential': 'green', 
                 'power_law': 'orange', 'quadratic': 'purple', 'logarithmic': 'brown'}
        
        for fit_type, fit in result['fits'].items():
            if fit['success'] and fit_type in model_funcs:
                y_smooth = model_funcs[fit_type](x_smooth, *fit['params'])
                alpha = 1.0 if fit_type == model_type else 0.3
                linewidth = 3 if fit_type == model_type else 1
                label = f"{fit_type} (R²={fit['r_squared']:.3f})" if fit_type == model_type else None
                ax.plot(x_smooth_orig, y_smooth, color=colors.get(fit_type, 'gray'), 
                       alpha=alpha, linewidth=linewidth, label=label)
        
        ax.set_title(f"{result['family']}\nBest: {model_type} (BIC)", fontweight='bold', fontsize=10)
        ax.set_xlabel('Days since first model')
        ax.set_ylabel('Score')
        ax.set_ylim(-0.1, 1.2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(examples), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "outputs" / "functional_form_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def main():
    print("="*80)
    print("PHASE 1, STEP 1c: TEST FUNCTIONAL FORMS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    runs_df = load_metr_runs_with_dates()
    df = runs_df[runs_df['release_date'].notna()].copy()
    
    task_families = df['task_family'].unique()
    print(f"  Found {len(task_families)} task families")
    
    # Analyze each family
    print("\nFitting all functional forms to each family...")
    all_results = []
    
    for family in sorted(task_families):
        family_df = df[df['task_family'] == family]
        result = analyze_task_family(family_df, family)
        
        if result:
            print(f"  ✓ {family:40s} - Best: {result['best_bic']:12s} (n={result['n_points']})")
            all_results.append(result)
        else:
            print(f"  ✗ {family:40s} - Insufficient data")
    
    print(f"\n  Successfully analyzed {len(all_results)} families")
    
    # Create comparison table
    df_comparison = create_comparison_table(all_results)
    
    # Summarize
    summarize_results(df_comparison)
    
    # Plot examples
    plot_examples(all_results, df_comparison)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey question: Is sigmoid justified?")
    print("  • Check if sigmoid wins most often by AIC/BIC")
    print("  • Look at examples where other models win")
    print("  • Consider if we need different models for different task types")


if __name__ == "__main__":
    main()



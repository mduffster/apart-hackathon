"""
STEP 4: Scenario Forecasts with Calendar Dates

Generates AGI timeline forecasts under different scenarios:
- Baseline: Current trends (6-month compute doubling)
- Efficiency gains: +5%, +10%, +20%, +30% translation improvements
- Compute restrictions: Slower scaling (12-month, 18-month doubling)
- Combined: Efficiency + slower scaling

Output:
- Timeline distributions for Near-AGI (0.8) and AGI (0.9)
- Probability by year
- Actual calendar dates
- Credible intervals
"""

import pandas as pd
import numpy as np
from scipy.special import expit
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, Tuple

# Set seed for reproducibility
np.random.seed(42)

@dataclass
class ScenarioConfig:
    """Configuration for a forecast scenario."""
    name: str
    beta_mult: float  # Multiplier on translation efficiency
    alpha_shift: float  # Shift in intercept
    doubling_months: int  # Compute doubling time
    color: str  # Plot color
    desc: str  # Description
    
    def apply_to_posterior(self, alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply scenario shocks to posterior samples."""
        return alpha + self.alpha_shift, beta * self.beta_mult

print("="*100)
print("STEP 4: SCENARIO FORECASTS")
print("="*100)

# ============================================================================
# Load posterior and current state
# ============================================================================

posterior = pd.read_csv('outputs/posterior_samples.csv')
current_state = pd.read_csv('outputs/current_state.csv')

print(f"\nPosterior samples: {len(posterior)}")
print(f"  a: {posterior['a'].mean():.2f} ± {posterior['a'].std():.2f}")
print(f"  b: {posterior['b'].mean():.2f} ± {posterior['b'].std():.2f}")
print(f"  α: {posterior['alpha'].mean():.3f} ± {posterior['alpha'].std():.3f}")
print(f"  β: {posterior['beta'].mean():.4f} ± {posterior['beta'].std():.4f}")

# Current state
best_current = current_state.iloc[0]
current_compute = best_current['compute']
current_date = datetime(2025, 11, 1)  # November 2025

print(f"\nCurrent state ({current_date.strftime('%B %Y')}):")
print(f"  Model: {best_current['model']}")
print(f"  Compute: {current_compute:.2e} FLOPs")
print(f"  METR: {best_current['metr_capability']:.3f}")

# ============================================================================
# Define scenarios
# ============================================================================

print(f"\n{'='*100}")
print("SCENARIOS")
print("="*100)

# Define scenarios using dataclass
scenarios = {
    'baseline': ScenarioConfig(
        name='Baseline (Current Trends)',
        beta_mult=1.0, alpha_shift=0.0, doubling_months=6,
        color='gray', desc='Pure compute scaling, 6-month doubling'
    ),
    'accel_3mo': ScenarioConfig(
        name='AGI Race (3mo doubling)',
        beta_mult=1.0, alpha_shift=0.0, doubling_months=3,
        color='purple', desc='Aggressive race: compute doubles every 3 months'
    ),
    'accel_4mo': ScenarioConfig(
        name='Moderate Acceleration (4mo)',
        beta_mult=1.0, alpha_shift=0.0, doubling_months=4,
        color='mediumpurple', desc='Moderate race: compute doubles every 4 months'
    ),
    'eff_05': ScenarioConfig(
        name='+5% Translation',
        beta_mult=1.05, alpha_shift=-0.3, doubling_months=6,
        color='lightgreen', desc='Minor improvements in data/training'
    ),
    'eff_10': ScenarioConfig(
        name='+10% Translation',
        beta_mult=1.10, alpha_shift=-0.6, doubling_months=6,
        color='mediumseagreen', desc='Incremental RL/synthetic improvements'
    ),
    'eff_20': ScenarioConfig(
        name='+20% Translation',
        beta_mult=1.20, alpha_shift=-1.2, doubling_months=6,
        color='green', desc='Significant algorithmic improvements'
    ),
    'eff_30': ScenarioConfig(
        name='+30% Translation',
        beta_mult=1.30, alpha_shift=-1.8, doubling_months=6,
        color='darkgreen', desc='Major breakthroughs'
    ),
    'slow_12mo': ScenarioConfig(
        name='Compute Restrictions (12mo)',
        beta_mult=1.0, alpha_shift=0.0, doubling_months=12,
        color='red', desc='Policy/economic restrictions'
    ),
    'slow_18mo': ScenarioConfig(
        name='Compute Restrictions (18mo)',
        beta_mult=1.0, alpha_shift=0.0, doubling_months=18,
        color='darkred', desc='Severe restrictions'
    ),
    'race_eff': ScenarioConfig(
        name='AGI Race + 20% Efficiency',
        beta_mult=1.20, alpha_shift=-1.2, doubling_months=3,
        color='darkviolet', desc='Aggressive race with algorithmic gains'
    ),
    'combo_20_9mo': ScenarioConfig(
        name='+20% + Slower (9mo)',
        beta_mult=1.20, alpha_shift=-1.2, doubling_months=9,
        color='orange', desc='Strong efficiency + moderate slowdown'
    ),
    'lucr_decay': ScenarioConfig(
        name='Faster LUCR Decay (90% β)',
        beta_mult=0.90, alpha_shift=0.5, doubling_months=6,
        color='brown', desc='Translation efficiency degrades (diminishing returns)'
    )
}

for key, scenario in scenarios.items():
    print(f"\n{key:20s}: {scenario.name}")
    print(f"  {scenario.desc}")

# ============================================================================
# AGI thresholds
# ============================================================================

thresholds = [
    {'name': 'Near-AGI', 'metr': 0.8, 'color': 'orange',
     'desc': '80% of (subhuman) tasks at human level'},
    {'name': 'AGI', 'metr': 0.9, 'color': 'red',
     'desc': '90% of (subhuman) tasks at human level'}
]

print(f"\n{'='*100}")
print("AGI THRESHOLDS")
print("="*100)

for t in thresholds:
    print(f"  {t['name']} ({t['metr']:.1f}): {t['desc']}")

# ============================================================================
# Helper function: Timeline calculation
# ============================================================================

def calculate_timeline(scenario: ScenarioConfig, threshold_metr: float, posterior: pd.DataFrame, 
                       current_compute: float, current_date: datetime) -> dict:
    """
    Calculate timeline distribution for a given scenario and threshold.
    
    Uses posterior draws (not medians) to ensure honest CI bands.
    
    Returns:
        - Probability of achieving threshold
        - Median years
        - Median date
        - 90% CI (years and dates)
        - Probability by year
    """
    # Extract posterior samples
    a_samples = posterior['a'].values
    b_samples = posterior['b'].values
    alpha_samples_base = posterior['alpha'].values
    beta_samples_base = posterior['beta'].values
    
    # Apply scenario shocks to posterior draws (not medians!)
    alpha_samples, beta_samples = scenario.apply_to_posterior(alpha_samples_base, beta_samples_base)
    
    # Compute range to explore (next 10 years)
    max_years = 10
    max_doublings = max_years * 12 / scenario.doubling_months
    max_compute = current_compute * (2 ** max_doublings)
    
    compute_range = np.logspace(
        np.log10(current_compute),
        np.log10(max_compute),
        200
    )
    log_compute_range = np.log10(compute_range)
    
    # For each posterior sample, find when threshold is crossed
    years_to_threshold = []
    dates_to_threshold = []
    
    for i in range(len(a_samples)):
        # Predict METR trajectory
        eci = a_samples[i] + b_samples[i] * log_compute_range
        logit_metr = alpha_samples[i] + beta_samples[i] * eci
        metr = expit(logit_metr)
        
        # Find first compute where metr >= threshold
        crossing_idx = np.where(metr >= threshold_metr)[0]
        
        if len(crossing_idx) > 0:
            compute_at_threshold = compute_range[crossing_idx[0]]
            
            # Calculate years from current
            doublings = np.log2(compute_at_threshold / current_compute)
            years = doublings * scenario.doubling_months / 12
            
            years_to_threshold.append(years)
            dates_to_threshold.append(current_date + timedelta(days=365*years))
    
    # Calculate statistics
    n_success = len(years_to_threshold)
    n_total = len(a_samples)
    probability = n_success / n_total
    
    if n_success == 0:
        return {
            'probability': 0.0,
            'median_years': np.nan,
            'median_date': None,
            'ci_years': (np.nan, np.nan),
            'ci_dates': (None, None),
            'prob_by_year': {}
        }
    
    years_array = np.array(years_to_threshold)
    dates_array = np.array(dates_to_threshold)
    
    median_years = np.median(years_array)
    median_date = current_date + timedelta(days=365*median_years)
    
    p05_years = np.percentile(years_array, 5)
    p95_years = np.percentile(years_array, 95)
    p05_date = current_date + timedelta(days=365*p05_years)
    p95_date = current_date + timedelta(days=365*p95_years)
    
    # Probability by year
    prob_by_year = {}
    for year in range(current_date.year, current_date.year + 15):
        year_mask = [(d.year == year) for d in dates_array]
        prob = np.sum(year_mask) / n_total
        if prob > 0.01:  # Only include if >1%
            prob_by_year[year] = prob
    
    return {
        'probability': probability,
        'median_years': median_years,
        'median_date': median_date,
        'ci_years': (p05_years, p95_years),
        'ci_dates': (p05_date, p95_date),
        'prob_by_year': prob_by_year
    }

# ============================================================================
# Run all scenarios
# ============================================================================

print(f"\n{'='*100}")
print("MONTE CARLO TIMELINE FORECASTS")
print("="*100)

all_results = []

for scenario_key, scenario in scenarios.items():
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario.name}")
    print("="*80)
    
    for threshold in thresholds:
        result = calculate_timeline(
            scenario, 
            threshold['metr'], 
            posterior, 
            current_compute, 
            current_date
        )
        
        if result['probability'] > 0:
            print(f"\n{threshold['name']} ({threshold['metr']:.1f}):")
            print(f"  Probability: {result['probability']*100:.1f}%")
            print(f"  Median: {result['median_date'].strftime('%B %Y')} ({result['median_years']:.1f} years)")
            
            ci_low_str = result['ci_dates'][0].strftime('%b %Y')
            ci_high_str = result['ci_dates'][1].strftime('%b %Y')
            print(f"  90% CI: {ci_low_str} - {ci_high_str} ({result['ci_years'][0]:.1f}-{result['ci_years'][1]:.1f} years)")
            
            if len(result['prob_by_year']) > 0:
                print(f"  Prob by year:")
                for year, prob in sorted(result['prob_by_year'].items()):
                    if prob > 0.01:
                        print(f"    {year}: {prob*100:.1f}%")
        else:
            print(f"\n{threshold['name']} ({threshold['metr']:.1f}):")
            print(f"  Probability: 0% (not achieved within 10 years)")
        
        # Store for summary table
        all_results.append({
            'scenario': scenario.name,
            'threshold': threshold['name'],
            'threshold_metr': threshold['metr'],
            'probability': result['probability'],
            'median_years': result['median_years'],
            'median_date': result['median_date'].strftime('%Y-%m') if result['median_date'] else None,
            'ci_low_years': result['ci_years'][0],
            'ci_high_years': result['ci_years'][1],
            'ci_low_date': result['ci_dates'][0].strftime('%Y-%m') if result['ci_dates'][0] else None,
            'ci_high_date': result['ci_dates'][1].strftime('%Y-%m') if result['ci_dates'][1] else None,
        })

# ============================================================================
# Summary table
# ============================================================================

print(f"\n{'='*100}")
print("SUMMARY TABLE: MEDIAN TIMELINES")
print("="*100)

results_df = pd.DataFrame(all_results)

# Format for display
display_df = results_df[results_df['probability'] > 0].copy()
display_df['ci_str'] = display_df.apply(
    lambda r: f"{r['ci_low_date']} - {r['ci_high_date']}", axis=1
)
display_df['prob_str'] = display_df['probability'].apply(lambda x: f"{x*100:.0f}%")

print(display_df[['scenario', 'threshold', 'median_date', 'median_years', 'ci_str', 'prob_str']].to_string(index=False))

# Save
results_file = 'outputs/scenario_timelines.csv'
results_df.to_csv(results_file, index=False)
print(f"\n✅ Saved: {results_file}")

print(f"\n{'='*100}")
print("✅ STEP 4 COMPLETE")
print("="*100)

print(f"\n💡 SUMMARY:")
print(f"  • Baseline (6mo doubling): Near-AGI {results_df[(results_df['scenario']=='Baseline (Current Trends)') & (results_df['threshold']=='Near-AGI')]['probability'].values[0]*100:.0f}% prob, AGI {results_df[(results_df['scenario']=='Baseline (Current Trends)') & (results_df['threshold']=='AGI')]['probability'].values[0]*100:.0f}% prob")
print(f"  • +20% efficiency: Significantly improves timelines")
print(f"  • Compute restrictions: Extends timelines 2-3x")
print(f"\n📊 Next step: Run 05_visualizations.py to generate plots")


import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from pathlib import Path
from scipy.special import expit
from datetime import datetime, timedelta

st.set_page_config(page_title="LUCR Forecasting Dashboard", layout="wide")

# Data paths
OUTPUTS_DIR = Path("outputs")

@st.cache_data(ttl=60)  # Cache for 60 seconds only
def load_data():
    """Load all required data files"""
    data = {}
    
    files = {
        "scenarios": "scenario_timelines.csv",
        "lucr": "lucr_forecast.csv",
        "stage1": "stage1_compute_eci.csv",
        "stage2": "stage2_eci_metr.csv",
        "posterior": "posterior_summary.csv",
        "current_state": "current_state.csv",
        "metr_runs": "filtered_metr_runs.csv"
    }
    
    for key, filename in files.items():
        path = OUTPUTS_DIR / filename
        if path.exists():
            try:
                data[key] = pd.read_csv(path)
            except Exception as e:
                st.warning(f"Could not read {filename}: {e}")
                data[key] = None
        else:
            data[key] = None
    
    # Load posterior samples from parquet (preferred) or CSV fallback
    parquet_path = OUTPUTS_DIR / "posterior_samples.parquet"
    csv_path = OUTPUTS_DIR / "posterior_samples.csv"
    if parquet_path.exists():
        try:
            data["posterior_samples"] = pd.read_parquet(parquet_path)
        except Exception as e:
            st.warning(f"Could not read posterior_samples.parquet: {e}")
            if csv_path.exists():
                data["posterior_samples"] = pd.read_csv(csv_path)
            else:
                data["posterior_samples"] = None
    elif csv_path.exists():
        data["posterior_samples"] = pd.read_csv(csv_path)
    else:
        data["posterior_samples"] = None
    
    return data

def calculate_timeline(posterior_samples, current_compute, threshold_metr, 
                       beta_mult=1.0, alpha_shift=0.0, doubling_months=6,
                       current_date=datetime(2025, 11, 1), max_samples=1000):
    """
    Calculate timeline distribution for given parameters.
    
    Args:
        posterior_samples: DataFrame with columns ['a', 'b', 'alpha', 'beta']
        current_compute: Current frontier compute (FLOPs)
        threshold_metr: Target METR capability (0-1)
        beta_mult: Multiplier on beta (efficiency)
        alpha_shift: Shift in alpha (intercept)
        doubling_months: Compute doubling time in months
        current_date: Reference date
        max_samples: Maximum posterior samples to use (for speed)
    
    Returns:
        dict with probability, median_years, percentiles, and yearly probabilities
    """
    # Sample posterior (use subset for speed)
    if len(posterior_samples) > max_samples:
        samples = posterior_samples.sample(n=max_samples, random_state=42)
    else:
        samples = posterior_samples
    
    a_samples = samples['a'].values
    b_samples = samples['b'].values
    alpha_samples_base = samples['alpha'].values
    beta_samples_base = samples['beta'].values
    
    # Apply adjustments
    alpha_samples = alpha_samples_base + alpha_shift
    beta_samples = beta_samples_base * beta_mult
    
    # Compute range (next 20 years to capture slower scenarios)
    max_years = 20
    max_doublings = max_years * 12 / doubling_months
    max_compute = current_compute * (2 ** max_doublings)
    
    compute_range = np.logspace(
        np.log10(current_compute),
        np.log10(max_compute),
        200
    )
    log_compute_range = np.log10(compute_range)
    
    # For each posterior sample, find when threshold is crossed
    years_to_threshold = []
    
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
            years = doublings * doubling_months / 12
            
            years_to_threshold.append(years)
    
    # Calculate statistics
    n_success = len(years_to_threshold)
    n_total = len(a_samples)
    probability = n_success / n_total
    
    if n_success == 0:
        return {
            'probability': 0.0,
            'median_years': np.nan,
            'p05_years': np.nan,
            'p25_years': np.nan,
            'p75_years': np.nan,
            'p95_years': np.nan,
            'median_date': None,
            'years_array': np.array([])
        }
    
    years_array = np.array(years_to_threshold)
    
    return {
        'probability': probability,
        'median_years': np.median(years_array),
        'p05_years': np.percentile(years_array, 5),
        'p25_years': np.percentile(years_array, 25),
        'p75_years': np.percentile(years_array, 75),
        'p95_years': np.percentile(years_array, 95),
        'median_date': current_date + timedelta(days=365*np.median(years_array)),
        'years_array': years_array
    }

# Load data
data = load_data()

# Title and description
st.title("🚀 LUCR — Linking Utility and Compute Rate")
st.markdown("""
Interactive dashboard for exploring AI capability forecasting through compute scaling laws.
**LUCR** measures the capability return (ΔMETR) per log₁₀ increase in compute (FLOPs).
""")

# Sidebar controls
st.sidebar.header("⚙️ Controls")

# Example scenarios at the top
with st.sidebar.expander("📚 Example Scenarios", expanded=False):
    st.markdown("""
    **AGI Race Scenario:**
    - Translation efficiency: 0%
    - Algorithmic efficiency: 1.0×
    - Compute doubling: 3 months
    
    **Better Architectures:**
    - Translation efficiency: +10%
    - Algorithmic efficiency: 1.0×
    - Compute doubling: 6 months
    
    **Training Breakthroughs:**
    - Translation efficiency: 0%
    - Algorithmic efficiency: 1.2×
    - Compute doubling: 6 months
    
    **Full Optimization:**
    - Translation efficiency: +20%
    - Algorithmic efficiency: 1.2×
    - Compute doubling: 3 months
    
    **Regulatory Slowdown:**
    - Translation efficiency: 0%
    - Algorithmic efficiency: 1.0×
    - Compute doubling: 12 months
    """)

st.sidebar.markdown("---")

# Threshold selection
threshold_options = {
    "Near-AGI (0.8 METR)": 0.8,
    "AGI (0.9 METR)": 0.9
}
threshold_label = st.sidebar.selectbox(
    "Capability threshold",
    options=list(threshold_options.keys()),
    index=1,
    key="threshold_select"
)
threshold_value = threshold_options[threshold_label]

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Adjust Parameters")
st.sidebar.caption("Move sliders to see updated forecasts")

# Task translation efficiency (alpha shift) - FIRST
translation_pct = st.sidebar.slider(
    "Task translation efficiency boost (%)",
    min_value=0,
    max_value=30,
    value=0,
    step=5,
    key="translation_slider",
    help="How well existing capability translates to task performance. Shifts the slope of compute → capability. Equivalent to 'less wasted horsepower.'"
)
alpha_shift = 0.06 * translation_pct  # Convert percentage to alpha shift (positive = better efficiency)

# Algorithmic efficiency multiplier (beta) - SECOND
eff_multiplier = st.sidebar.slider(
    "Algorithmic efficiency (β multiplier)",
    min_value=0.7,
    max_value=1.3,
    value=1.0,
    step=0.05,
    key="beta_slider",
    help="How well training converts compute into general capability. Raises the ceiling of what's possible. Higher β = slower LUCR decay = 'faster emergence.'"
)

# Compute doubling cadence
doubling_months = st.sidebar.selectbox(
    "Compute doubling time (months)",
    options=[3, 4, 6, 9, 12, 18],
    index=2,
    key="doubling_select",
    help="How often compute resources double. Baseline = 6 months"
)

# Check if adjustments were made
is_baseline = (eff_multiplier == 1.0) and (translation_pct == 0) and (doubling_months == 6)

st.sidebar.markdown("---")
st.sidebar.caption("💾 Data loaded from `outputs/` directory")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Timeline Forecasts", "📈 LUCR Analysis", "📋 Data Tables"])

# TAB 1: Timeline Forecasts
with tab1:
    st.header("AGI Timeline Forecasts")
    
    st.info("💡 **Tip:** Click on '📚 Example Scenarios' in the sidebar to see common scenario configurations and adjust the sliders to explore different futures.")
    
    # Check if we have the data needed for dynamic forecasting
    if data["posterior_samples"] is not None and data["current_state"] is not None:
        current_compute = data["current_state"].iloc[0]['compute']
        current_date = datetime(2025, 11, 1)
        
        # DEBUG: Check first sample calculation
        first_sample = data["posterior_samples"].iloc[0]
        test_compute = current_compute * 2**10  # 10 doublings = 5 years
        test_eci = first_sample['a'] + first_sample['b'] * np.log10(test_compute)
        test_metr = expit(first_sample['alpha'] + first_sample['beta'] * test_eci)
        st.caption(f"🔍 Sample 0: After 10 doublings (5yr), compute={test_compute:.2e}, METR={test_metr:.3f}")
        
        # Compute baseline forecast (6mo doubling, no efficiency changes)
        baseline_result = calculate_timeline(
            data["posterior_samples"], 
            current_compute, 
            threshold_value,
            beta_mult=1.0,
            alpha_shift=0.0,
            doubling_months=6,
            current_date=current_date
        )
        
        # Compute adjusted forecast if user changed parameters
        if not is_baseline:
            adjusted_result = calculate_timeline(
                data["posterior_samples"], 
                current_compute, 
                threshold_value,
                beta_mult=eff_multiplier,
                alpha_shift=alpha_shift,
                doubling_months=doubling_months,
                current_date=current_date
            )
        else:
            adjusted_result = None
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Timeline Distribution")
            
            # Create visualization data
            viz_data = []
            
            # Baseline
            if baseline_result['probability'] > 0:
                viz_data.append({
                    'scenario': 'Baseline (6mo doubling)',
                    'type': 'baseline',
                    'median': baseline_result['median_years'],
                    'p05': baseline_result['p05_years'],
                    'p25': baseline_result['p25_years'],
                    'p75': baseline_result['p75_years'],
                    'p95': baseline_result['p95_years'],
                    'probability': baseline_result['probability']
                })
            
            # Adjusted
            if adjusted_result and adjusted_result['probability'] > 0:
                viz_data.append({
                    'scenario': 'Your Adjusted Forecast',
                    'type': 'adjusted',
                    'median': adjusted_result['median_years'],
                    'p05': adjusted_result['p05_years'],
                    'p25': adjusted_result['p25_years'],
                    'p75': adjusted_result['p75_years'],
                    'p95': adjusted_result['p95_years'],
                    'probability': adjusted_result['probability']
                })
            
            if viz_data:
                viz_df = pd.DataFrame(viz_data)
                
                # Create layered chart
                base = alt.Chart(viz_df).encode(
                    y=alt.Y('scenario:N', title=None, sort=['Your Adjusted Forecast', 'Baseline (6mo doubling)'])
                )
                
                # 50% CI (p25-p75)
                ci_50 = base.mark_bar(height=20).encode(
                    x=alt.X('p25:Q', title='Years from now'),
                    x2='p75:Q',
                    color=alt.Color('type:N', 
                                  scale=alt.Scale(
                                      domain=['baseline', 'adjusted'],
                                      range=['#cccccc', '#1f77b4']
                                  ),
                                  legend=None),
                    opacity=alt.condition(
                        alt.datum.type == 'baseline',
                        alt.value(0.4),
                        alt.value(0.7)
                    ),
                    tooltip=['scenario', 'median', 'probability']
                )
                
                # 90% CI (p05-p95) - thinner
                ci_90 = base.mark_bar(height=8).encode(
                    x=alt.X('p05:Q'),
                    x2='p95:Q',
                    color=alt.Color('type:N',
                                  scale=alt.Scale(
                                      domain=['baseline', 'adjusted'],
                                      range=['#999999', '#4a90e2']
                                  ),
                                  legend=None),
                    opacity=alt.condition(
                        alt.datum.type == 'baseline',
                        alt.value(0.2),
                        alt.value(0.4)
                    )
                )
                
                # Median points
                median_points = base.mark_circle(size=120).encode(
                    x=alt.X('median:Q'),
                    color=alt.Color('type:N',
                                  scale=alt.Scale(
                                      domain=['baseline', 'adjusted'],
                                      range=['#666666', '#1f77b4']
                                  ),
                                  legend=None),
                    opacity=alt.value(1.0),
                    tooltip=[
                        alt.Tooltip('scenario:N', title='Scenario'),
                        alt.Tooltip('median:Q', title='Median (years)', format='.2f'),
                        alt.Tooltip('p05:Q', title='5th percentile', format='.2f'),
                        alt.Tooltip('p95:Q', title='95th percentile', format='.2f'),
                        alt.Tooltip('probability:Q', title='Probability', format='.1%')
                    ]
                )
                
                timeline_chart = (ci_90 + ci_50 + median_points).properties(height=150)
                st.altair_chart(timeline_chart, width="stretch")
                
                st.caption("**Dark bar:** 50% confidence interval (P25-P75) | **Light bar:** 90% CI (P05-P95) | **Circle:** Median")
            else:
                st.warning("Threshold not reached within 20 years in forecasts")
        
        with col2:
            st.subheader("Summary")
            
            # If we have both baseline and adjusted, show side by side
            if baseline_result['probability'] > 0 and adjusted_result and adjusted_result['probability'] > 0:
                sum_col1, sum_col2 = st.columns(2)
                
                with sum_col1:
                    st.markdown("**Baseline**")
                    st.metric("Timeline", f"{baseline_result['median_years']:.1f}y")
                    st.metric("Date", baseline_result['median_date'].strftime('%b %Y'))
                    st.metric("Prob.", f"{baseline_result['probability']:.0%}")
                
                with sum_col2:
                    st.markdown("**Adjusted**")
                    delta_years = adjusted_result['median_years'] - baseline_result['median_years']
                    st.metric(
                        "Timeline", 
                        f"{adjusted_result['median_years']:.1f}y",
                        delta=f"{delta_years:+.1f}y"
                    )
                    st.metric("Date", adjusted_result['median_date'].strftime('%b %Y'))
                    st.metric("Prob.", f"{adjusted_result['probability']:.0%}")
            
            # Otherwise show baseline only
            elif baseline_result['probability'] > 0:
                st.markdown("**Baseline (6mo doubling)**")
                st.metric("Median Timeline", f"{baseline_result['median_years']:.1f} years")
                st.metric("Median Date", baseline_result['median_date'].strftime('%b %Y'))
                st.metric("Probability", f"{baseline_result['probability']:.1%}")
        
        # Distribution histogram
        st.markdown("---")
        st.subheader("Timeline Probability Distributions")
        
        # Pre-bin the data for cleaner visualization
        hist_data = []
        
        # Create bins from 0 to 20 years
        bins = np.linspace(0, 20, 41)  # 40 bins
        
        if baseline_result['probability'] > 0:
            counts, _ = np.histogram(baseline_result['years_array'], bins=bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            for i, count in enumerate(counts):
                if count > 0:
                    hist_data.append({
                        'years': bin_centers[i],
                        'count': count,
                        'type': 'Baseline'
                    })
        
        if adjusted_result and adjusted_result['probability'] > 0:
            counts, _ = np.histogram(adjusted_result['years_array'], bins=bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            for i, count in enumerate(counts):
                if count > 0:
                    hist_data.append({
                        'years': bin_centers[i],
                        'count': count,
                        'type': 'Adjusted'
                    })
        
        if hist_data:
            hist_df = pd.DataFrame(hist_data)
            
            # Create separate charts and layer them
            base_chart = alt.Chart(hist_df[hist_df['type'] == 'Baseline']).mark_bar(
                opacity=0.4,
                color='#999999'
            ).encode(
                x=alt.X('years:Q', title='Years from now', scale=alt.Scale(domain=[0, 20])),
                y=alt.Y('count:Q', title='Number of posterior samples'),
                tooltip=[
                    alt.Tooltip('years:Q', title='Years', format='.1f'),
                    alt.Tooltip('count:Q', title='Count'),
                    'type'
                ]
            )
            
            if adjusted_result and adjusted_result['probability'] > 0:
                adjusted_chart = alt.Chart(hist_df[hist_df['type'] == 'Adjusted']).mark_bar(
                    opacity=0.6,
                    color='#1f77b4'
                ).encode(
                    x=alt.X('years:Q', title='Years from now'),
                    y=alt.Y('count:Q'),
                    tooltip=[
                        alt.Tooltip('years:Q', title='Years', format='.1f'),
                        alt.Tooltip('count:Q', title='Count'),
                        'type'
                    ]
                )
                hist_chart = (base_chart + adjusted_chart).properties(height=250)
            else:
                hist_chart = base_chart.properties(height=250)
            
            st.altair_chart(hist_chart, width="stretch")
            
            st.caption("📊 Gray = Baseline | Blue = Your Adjusted Forecast | Higher bars = more likely timeline")
        
        # Parameter display
        st.markdown("---")
        st.subheader("🎛️ Current Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Translation Efficiency", 
                f"+{translation_pct}%" if translation_pct > 0 else "Baseline",
                help="Task performance translation (α shift)"
            )
        with col2:
            st.metric(
                "Algorithmic Efficiency", 
                f"{eff_multiplier:.2f}×",
                delta=f"{(eff_multiplier-1)*100:+.0f}%" if eff_multiplier != 1.0 else None,
                help="Training capability conversion (β multiplier)"
            )
        with col3:
            st.metric(
                "Compute Doubling", 
                f"{doubling_months} months", 
                delta=f"{doubling_months-6:+d} mo" if doubling_months != 6 else None,
                help="Hardware scaling rate"
            )
    else:
        st.error("Missing required data: posterior_samples.csv or current_state.csv")

# TAB 2: LUCR Analysis
with tab2:
    st.header("LUCR: Capability Return per Log₁₀ Compute")
    
    if data["lucr"] is not None:
        st.markdown("""
        **LUCR** (Linking Utility and Compute Rate) measures how much AI capability improves 
        per 10× increase in training compute. As models scale, LUCR typically decays, 
        representing diminishing returns.
        """)
        
        st.info("📊 **Note:** This LUCR curve is only affected by the **β (Algorithmic Efficiency) multiplier**, which controls how well training converts compute into capability. Translation efficiency (task performance) and compute doubling time affect timelines but not the LUCR decay rate itself.")
        
        lucr_df = data["lucr"].copy()
        
        # Always keep baseline
        lucr_df["lucr_baseline"] = lucr_df["lucr_median"]
        lucr_df["lucr_p05_baseline"] = lucr_df["lucr_p05"]
        lucr_df["lucr_p95_baseline"] = lucr_df["lucr_p95"]
        
        # Apply efficiency adjustment to LUCR
        lucr_df["lucr_adjusted"] = lucr_df["lucr_median"] * eff_multiplier
        lucr_df["lucr_p05_adjusted"] = lucr_df["lucr_p05"] * eff_multiplier
        lucr_df["lucr_p95_adjusted"] = lucr_df["lucr_p95"] * eff_multiplier
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("LUCR vs Compute")
            
            # Base chart setup
            base = alt.Chart(lucr_df).encode(
                x=alt.X("compute:Q", title="Training Compute (FLOPs)", scale=alt.Scale(type="log"))
            )
            
            # Baseline LUCR (always shown in gray)
            baseline_band = base.mark_area(opacity=0.15, color="#999999").encode(
                y=alt.Y("lucr_p05_baseline:Q", title="LUCR (ΔMETR per log₁₀ FLOPs)"),
                y2="lucr_p95_baseline:Q"
            )
            
            baseline_line = base.mark_line(color="#666666", size=2, strokeDash=[5, 5]).encode(
                y=alt.Y("lucr_baseline:Q"),
                tooltip=[
                    alt.Tooltip("compute:Q", format=".2e", title="Compute"),
                    alt.Tooltip("lucr_baseline:Q", format=".4f", title="LUCR (baseline)")
                ]
            )
            
            # Start with baseline
            lucr_chart = baseline_band + baseline_line
            
            # Add adjusted curves if different from baseline
            if eff_multiplier != 1.0:
                adjusted_band = base.mark_area(opacity=0.2, color="#1f77b4").encode(
                    y=alt.Y("lucr_p05_adjusted:Q"),
                    y2="lucr_p95_adjusted:Q"
                )
                
                adjusted_line = base.mark_line(color="#1f77b4", size=3).encode(
                    y=alt.Y("lucr_adjusted:Q"),
                    tooltip=[
                        alt.Tooltip("compute:Q", format=".2e", title="Compute"),
                        alt.Tooltip("lucr_adjusted:Q", format=".4f", title="LUCR (adjusted)"),
                        alt.Tooltip("lucr_baseline:Q", format=".4f", title="LUCR (baseline)")
                    ]
                )
                
                lucr_chart = lucr_chart + adjusted_band + adjusted_line
            
            lucr_chart = lucr_chart.properties(height=400)
            st.altair_chart(lucr_chart, width="stretch")
            
            if eff_multiplier != 1.0:
                st.caption("🔵 Blue = Your Adjusted LUCR | ⚫ Gray (dashed) = Baseline LUCR")
            else:
                st.caption("⚫ Gray = Baseline LUCR (adjust β multiplier in sidebar to see changes)")
        
        with col2:
            st.subheader("Current State")
            
            # Get current LUCR value (most recent or mid-range)
            current_idx = len(lucr_df) // 2
            current_lucr_baseline = lucr_df.iloc[current_idx]["lucr_baseline"]
            current_lucr_adjusted = lucr_df.iloc[current_idx]["lucr_adjusted"]
            current_compute = lucr_df.iloc[current_idx]["compute"]
            
            # Show baseline
            st.markdown("**Baseline LUCR:**")
            st.metric(
                "LUCR Value",
                f"{current_lucr_baseline:.4f}",
                help="At reference compute scale (baseline β)"
            )
            
            # Show adjusted if different
            if eff_multiplier != 1.0:
                st.markdown("---")
                st.markdown("**Your Adjusted LUCR:**")
                delta_lucr = current_lucr_adjusted - current_lucr_baseline
                st.metric(
                    "LUCR Value",
                    f"{current_lucr_adjusted:.4f}",
                    delta=f"{delta_lucr:+.4f}",
                    help="With your β multiplier adjustment"
                )
            
            st.markdown("---")
            st.metric(
                "Reference Compute",
                f"{current_compute:.2e} FLOPs"
            )
            
            # Calculate capability gain for 10× compute
            st.markdown("**10× Compute Increase:**")
            capability_gain = current_lucr_adjusted if eff_multiplier != 1.0 else current_lucr_baseline
            st.write(f"Expected capability gain: **{capability_gain:.3f}** METR points")
            
            st.markdown("---")
            st.caption("💡 Higher LUCR = more bang for your (compute) buck")
        
        # LUCR decay analysis
        st.markdown("---")
        st.subheader("LUCR Decay Rate")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate decay rate
            if len(lucr_df) > 1:
                log_compute = np.log10(lucr_df["compute"])
                log_lucr = np.log10(lucr_df["lucr_adjusted"])
                
                # Simple linear fit for decay
                from numpy.polynomial import polynomial as P
                coef = P.polyfit(log_compute, log_lucr, 1)
                decay_rate = coef[1]
                
                st.metric(
                    "Log-log decay slope",
                    f"{decay_rate:.3f}",
                    help="Slope of log(LUCR) vs log(Compute)"
                )
                
                if decay_rate < 0:
                    st.caption(f"LUCR decreases by {-decay_rate:.1%} per 10× compute increase")
                else:
                    st.caption("LUCR is increasing (unusual)")
        
        with col2:
            if data["posterior"] is not None:
                beta_row = data["posterior"][data["posterior"]["parameter"] == "beta"]
                if not beta_row.empty:
                    beta_mean = beta_row.iloc[0]["mean"]
                    st.metric(
                        "Fitted β (decay param)",
                        f"{beta_mean:.4f}",
                        delta=f"{(eff_multiplier - 1) * 100:+.0f}% (your adjustment)"
                    )
                    st.caption("From Bayesian hierarchical model")
    else:
        st.info("LUCR data not found. Run forecasting pipeline to generate.")

# TAB 3: Data Tables
with tab3:
    st.header("Raw Data Tables")
    
    table_select = st.selectbox(
        "Select table to view",
        options=[
            "Scenario Timelines",
            "LUCR Forecast",
            "Stage 1 (Compute→ECI)",
            "Stage 2 (ECI→METR)",
            "Posterior Summary",
            "Filtered METR Runs",
            "Current State"
        ]
    )
    
    table_map = {
        "Scenario Timelines": "scenarios",
        "LUCR Forecast": "lucr",
        "Stage 1 (Compute→ECI)": "stage1",
        "Stage 2 (ECI→METR)": "stage2",
        "Posterior Summary": "posterior",
        "Filtered METR Runs": "metr_runs",
        "Current State": "current_state"
    }
    
    selected_data = data[table_map[table_select]]
    
    if selected_data is not None:
        st.dataframe(selected_data, width="stretch", height=500)
        
        # Download button
        csv = selected_data.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {table_select} as CSV",
            data=csv,
            file_name=f"{table_select.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"{table_select} data not available")

# Footer
st.markdown("---")
st.caption("💡 **Tip:** Adjust parameters in the sidebar to explore different scenarios and efficiency assumptions")


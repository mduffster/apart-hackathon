# LUCR Forecasting Dashboard

Interactive Streamlit app for exploring AI capability forecasting through compute scaling laws with real-time parameter adjustments.

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### 📊 Timeline Forecasts (Tab 1)
- **Dynamic baseline vs adjusted forecasting** - See how your parameter changes affect AGI timelines
- **Real-time computation** using Bayesian posterior samples
- **Visual comparison** with confidence intervals (50% and 90% CI)
- **Timeline probability distributions** showing likelihood across years
- **Interactive threshold selection** (Near-AGI at 0.8, AGI at 0.9 METR capability)

When you adjust parameters in the sidebar, the app:
- Always shows the **baseline** (gray, 6-month doubling, no efficiency changes)
- Computes and overlays your **adjusted forecast** (blue) in real-time
- Shows the median timeline shift and probability changes

### 📈 LUCR Analysis (Tab 2)
- **Baseline vs adjusted LUCR curves** showing capability returns per 10× compute
- **Confidence bands** from Bayesian posterior (5th to 95th percentile)
- **LUCR decay analysis** with log-log slopes
- **Only affected by β (algorithmic efficiency) multiplier** - other parameters affect timelines but not LUCR

The LUCR curve visualizes how much AI capability (ΔMETR) you gain per 10× increase in training compute, and how this decays as models scale.

### 📋 Data Tables (Tab 3)
- Browse all underlying data files
- Download any table as CSV
- Includes scenario timelines, LUCR forecasts, posterior samples, METR runs, etc.

## Interactive Controls

### Sidebar Parameters

**📚 Example Scenarios** (expandable at top)
- AGI Race: 3-month doubling, baseline efficiency
- Better Architectures: +10% task translation efficiency
- Training Breakthroughs: 1.2× algorithmic efficiency
- Full Optimization: +20% task translation + 1.2× algorithmic + 3mo doubling
- Regulatory Slowdown: 12-month doubling

**Threshold Selection**
- Near-AGI (0.8 METR): 80% of subhuman tasks at human level
- AGI (0.9 METR): 90% of subhuman tasks at human level

**Adjust Parameters:**
- **Task Translation Efficiency boost** (0% - 30%)
  - How well existing capability translates to task performance
  - Shifts the **slope** of compute → capability mapping
  - Equivalent to "less wasted horsepower"
  - Each +10% ≈ +0.6 shift in α parameter
  - Affects timelines but NOT LUCR curves

- **Algorithmic Efficiency (β) multiplier** (0.7 - 1.3)
  - How well training converts compute into general capability
  - Raises the **ceiling** of what's possible
  - Higher β = slower LUCR decay = "faster emergence"
  - Affects both timelines AND LUCR curves
  - Baseline = 1.0

- **Compute doubling time** (3, 4, 6, 9, 12, or 18 months)
  - How often compute resources double (hardware scaling)
  - Faster doubling = earlier timelines
  - Baseline = 6 months

### How Parameters Affect Forecasts

| Parameter | Meaning | Affects Timelines | Affects LUCR Curve |
|-----------|---------|------------------|--------------------|
| **Task translation efficiency** | Task performance from capability | ✅ Yes | ❌ No |
| **Algorithmic efficiency** | Capability from training compute | ✅ Yes | ✅ Yes |
| **Compute doubling** | Hardware scaling rate | ✅ Yes | ❌ No |

**Key distinction:**
- **Algorithmic efficiency**: Raises the ceiling ("faster emergence")
- **Task translation efficiency**: Shifts the slope ("less wasted horsepower")

## Data Requirements

The app loads data from the `outputs/` directory. Required files:
- `posterior_samples.csv` - Bayesian posterior samples for dynamic forecasting (**required**)
- `current_state.csv` - Current frontier model state (**required**)
- `lucr_forecast.csv` - Pre-computed LUCR projections
- `posterior_summary.csv` - Parameter summary statistics
- `stage1_compute_eci.csv` - Compute to ECI mapping (optional)
- `stage2_eci_metr.csv` - ECI to METR mapping (optional)
- `scenario_timelines.csv` - Pre-computed scenario results (optional)
- `filtered_metr_runs.csv` - METR benchmark data (optional)

## How It Works

This dashboard uses a **Bayesian hierarchical model** that:

1. **Stage 1**: Maps training compute (FLOPs) → Epoch Capability Index (ECI)
   - Based on historical AI models and Kaplan scaling laws
   
2. **Stage 2**: Maps ECI → METR benchmark performance (0-1 capability score)
   - Links capability indices to concrete task performance
   
3. **Forecasting**: Projects timelines under different scenarios
   - Uses posterior samples to capture uncertainty
   - Applies user adjustments to parameters in real-time
   - Computes when capability thresholds are reached

**LUCR** (Linking Utility and Compute Rate) measures how much AI capability (ΔMETR) improves per 10× increase in training compute, accounting for diminishing returns as models scale.

## Tips for Use

1. **Start with example scenarios** - Click "📚 Example Scenarios" in the sidebar to see common configurations
2. **Try the AGI Race scenario** - Set compute doubling to 3 months and watch timelines shift
3. **Explore efficiency gains** - Increase β multiplier to 1.2 to see sustained algorithmic improvements
4. **Compare visually** - The baseline (gray) always stays visible for comparison
5. **Check both tabs** - Timeline Forecasts shows when AGI arrives; LUCR Analysis shows why

## Technical Details

- **Forecast horizon**: 20 years from November 2025
- **Posterior samples**: Uses 1,000 samples for speed (configurable)
- **Confidence intervals**: 50% (P25-P75) and 90% (P05-P95)
- **Probability calculations**: Fraction of posterior samples reaching threshold within timeframe


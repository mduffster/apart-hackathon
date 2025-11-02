# AGI Timeline Forecasting: Hierarchical Bayesian Analysis

**Interactive Streamlit app + complete analysis pipeline for forecasting AGI timelines through compute scaling laws and capability benchmarks.**

---

## 🚀 Quick Start: Interactive App

**Try the interactive dashboard to explore different scenarios:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

Visit `http://localhost:8501` to:
- Adjust parameters and see timelines update in real-time
- Compare baseline vs custom scenarios
- Explore LUCR (capability decay) curves
- Download all underlying data

See [README_APP.md](README_APP.md) for full app documentation.

---

## 📊 Overview

This project provides a **hierarchical Bayesian forecast** of AGI timelines based on:
1. **METR task performance** (subhuman runs only - Method B filtering)
2. **Epoch AI compute data** (~80 models)
3. **Two-stage model**: Compute → ECI → METR

### 🎯 Key Contribution: LUCR (Linking Utility and Compute Rate)

**LUCR** decomposes AGI progress into two components:
1. **Scaling Efficiency** (dECI/d log C): How much capability gain per 10× compute
2. **Translation Efficiency** (dMETR/dECI): How effectively capability translates to real-world tasks

**Key Finding**: With subhuman-task filtering, LUCR analysis reveals:
- **We're still BEFORE the efficiency peak** - in the ascending phase
- Current METR (0.22) shows we're very early on hard tasks
- Reaching human-level requires both massive compute scaling AND sustained efficiency
- Pure compute scaling alone shows limited AGI probability within 10 years

---

## 📈 Key Findings

### Current State
Using **Method B filtering** (only including runs where models are below human performance):
- **Current frontier METR: 0.22** (Claude 3.7 Sonnet on subhuman tasks)
- This is much lower than the 0.62 we'd get including solved tasks
- **The "catching up to humans" problem is real**

### Baseline Timeline (6-month compute doubling)
- **Near-AGI (0.8 METR)**: ~14.7 years (2040), ~80% probability
- **AGI (0.9 METR)**: ~16.5 years (2042), ~65% probability

### Efficiency Scenarios

| Scenario | Near-AGI (0.8) | AGI (0.9) |
|----------|----------------|-----------|
| Baseline | 9.3 years | 9.5 years |
| AGI Race (3mo doubling) | 7.4 years | 8.2 years |
| +20% Translation | 8.2 years | 8.8 years |
| +30% Translation | 7.3 years | 8.2 years |
| Regulatory Slowdown (12mo) | >10 years | > 10 years |

### Frontier Scaling Shows Diminishing Returns
- **GPT-3 era** (10²²-10²⁴ FLOPs): +16.7 ECI per 10× compute
- **Modern era** (≥10²⁴ FLOPs): +7.2 ECI per 10× compute (**43% less efficient**)
- Suggests we may be experiencing algorithmic efficiency decay at the frontier, though LUCR indicates there is still growth to be found

---

## 🔧 Analysis Pipeline

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyMC (Bayesian modeling)
- ArviZ (MCMC diagnostics)
- Pandas, NumPy, SciPy
- Matplotlib, Seaborn
- Streamlit, Altair (for app)

### Pipeline Steps

#### 1. Filter METR Data (`01_filter_subhuman_metr.py`)
- Load 18,964 METR runs
- Filter for 8 well-covered models (≥10 tasks each)
- **Method B**: Keep only runs where model < human (4,490 runs, 69%)
- **Frozen cohort**: Fix task families and weights (November 2025)
  - Prevents data leakage
  - Ensures consistent evaluation across models
- **Headroom/CGI metrics**: Track progress toward human-level
  - CGI (Capability Gain Index) = METR capability
  - Headroom = 1 - CGI (room remaining to 1.0)
- **Output**: `outputs/filtered_metr_runs.csv`, current frontier METR = 0.22

#### 2. Build Datasets (`02_compute_eci_metr_data.py`)
- **Stage 1**: 80 Epoch models with Compute + ECI
- **Stage 2**: 7 overlap models with ECI + METR
  - 4 from Epoch (GPT-4, GPT-4o, o1-preview, GPT-4 Turbo)
  - 3 estimated (Claude models, ECI from compute relationship)
- **Output**: `outputs/stage1_compute_eci.csv`, `outputs/stage2_eci_metr.csv`

#### 3. Bayesian Model (`03_bayesian_hierarchical_model.py`)
- **Stage 1**: C → ECI (era-stratified with partial pooling, 80 models)
  - ECI = a + b × log₁₀(C) + γ_era[era]
  - Global posterior: a = -26 ± 27, b = 6.1 ± 1.1
  - Era adjustments capture regime shifts
  - **Student-t likelihoods** (ν=5) for outlier resistance
  
- **Stage 2**: ECI → METR (logit link, 7 models + 10 pseudo-obs)
  - logit(METR) = α + β × ECI
  - Posterior: α = -13.7 ± 0.9, **β = 0.081 ± 0.007**
  - **Student-t likelihoods** (ν=5) for robustness
  
- **Posterior Predictive Checks**:
  - Model calibration validated
  - Samples saved for downstream use
  
- **LUCR Decomposition**:
  - Scaling efficiency (dECI/d log C): 6.1 ± 1.1
  - Translation efficiency (dMETR/dECI): 0.0018 ± 0.0001
  - Combined (dMETR/d log C): 0.011 ± 0.002

- **Output**: `outputs/posterior_samples.csv`, `outputs/posterior_summary.csv`

#### 4. Scenario Forecasts (`04_scenario_forecasts.py`)
- 12 scenarios: baseline, efficiency improvements, compute restrictions, combinations
- 2 thresholds: Near-AGI (0.8), AGI (0.9)
- Monte Carlo over 15,000+ posterior samples
- **Output**: `outputs/scenario_timelines.csv` with calendar dates and probabilities

#### 5. Visualizations (`05_visualizations.py`)
- Data overview (filtering, model fits, LUCR)
- Scenario timelines (probabilities, error bars)
- **Output**: `outputs/01_data_overview.png`, `outputs/02_scenario_timelines.png`

#### 6. Scenario Analysis (`06_scenario_analysis.py`)
- Comprehensive scenario comparison
- Probability distributions by year
- **Output**: `outputs/03_scenario_analysis_comprehensive.png`, `outputs/04_key_scenarios.png`, `outputs/05_probability_analysis.png`

#### 7. GPT-2 Sensitivity (`07_gpt2_sensitivity.py`)
- Analysis of pre-RLHF era impact
- Era-stratified scaling analysis
- **Output**: `outputs/06_gpt2_sensitivity.png`, `outputs/gpt2_sensitivity_results.csv`

#### 8. LUCR Main Chart (`08_lucr_main_chart.py`)
**⭐ Main Analysis Chart**

Comprehensive LUCR visualization highlighting the key contribution:
- LUCR curve over compute (showing efficiency dynamics)
- Forecast trajectories with uncertainty
- Parameter distributions

**Output**: `outputs/09_lucr_main_chart.png`

### Running the Complete Pipeline

```bash
# Run all steps in sequence
python 00_impute_compute.py
python 01_filter_subhuman_metr.py
python 02_compute_eci_metr_data.py
python 03_bayesian_hierarchical_model.py
python 04_scenario_forecasts.py
python 05_visualizations.py
python 06_scenario_analysis.py
python 07_gpt2_sensitivity.py
python 08_lucr_main_chart.py

# Launch interactive app
streamlit run app.py
```

Each step saves outputs that the next step loads. Total runtime: ~5-10 minutes (excluding MCMC sampling).

---

## 📂 Project Structure

```
├── app.py                          # Streamlit interactive dashboard
├── requirements.txt                # All dependencies (pipeline + app)
├── README.md                       # This file
├── README_APP.md                   # App documentation
│
├── 00_impute_compute.py           # Impute missing compute values
├── 01_filter_subhuman_metr.py     # METR data filtering
├── 02_compute_eci_metr_data.py    # Build datasets
├── 03_bayesian_hierarchical_model.py  # Bayesian MCMC
├── 04_scenario_forecasts.py       # Generate timelines
├── 05_visualizations.py           # Create plots
├── 06_scenario_analysis.py        # Scenario comparison
├── 07_gpt2_sensitivity.py         # Era analysis
├── 08_lucr_main_chart.py          # Main LUCR chart
│
├── data/
│   ├── ai_models/                 # Epoch AI model data
│   └── METR/                      # METR benchmark results
│
├── outputs/                       # All generated outputs
│   ├── *.csv                      # Data outputs
│   ├── *.png                      # Visualizations
│   └── *.parquet                  # Posterior samples
│
└── archive/                       # Previous exploration scripts
```

---

## 🔬 Methodology

### Method B Filtering

**Why subhuman tasks only?**

We tested 3 approaches:
- **Method 0** (Current): Exclude trivial tasks only → METR = 0.62
- **Method A**: Exclude task families where any model ≥ human → METR = 0.13 (too restrictive)
- **Method B**: Run-level filter (model < human on each task) → **METR = 0.22** ✓

**Method B is best because:**
1. Focuses on the "catching up to humans" problem
2. Still uses 69% of runs (vs 24% for Method A)
3. Honest about remaining progress needed
4. Avoids conflating "solved trivial tasks" with "near AGI"

### Statistical Robustness

- **Student-t likelihoods** (ν=5) handle outliers without distorting fit
- **Era stratification** (Pre-RLHF, GPT-3, Modern) with partial pooling captures regime shifts
- **Posterior predictive checks** confirm model calibration
- **Frozen cohort weights** (Nov 2025) prevent data leakage and ensure reproducibility
- **Headroom/CGI metrics** provide interpretable progress tracking

### Key Uncertainties

- Translation efficiency (β) is based on only 7 models (but robust with Student-t)
- Assumption that subhuman tasks will reach 0.9 (may require breakthroughs)
- Current compute estimates for Claude/o1 are approximate
- Era-specific slopes suggest regime changes - extrapolation to frontier uncertain

---

## 🎯 Key Insights

**Models have solved many easy tasks but struggle on hard ones:**
- 38 of 50 task families (76%) have superhuman performance
- But on the remaining **subhuman tasks, best model is only 22% toward human**
- This suggests significant **architectural/algorithmic gaps**, not just compute

**Timelines are longer than naive scaling suggests:**
- Pure compute scaling → AGI by 2035 (baseline)
- Needs +20-30% efficiency gains for AGI by 2033-2034
- Under compute restrictions (12mo doubling), AGI unlikely within 20 years

**Translation efficiency is the bottleneck:**
- β (ECI → METR) = 0.081 ± 0.007
- This tight posterior suggests slow capability→performance translation
- Improving β by 20% accelerates timelines by ~1-2 years

---

## 📊 Interactive App Features

The Streamlit dashboard (`app.py`) provides:

### Timeline Forecasts Tab
- Dynamic baseline vs adjusted forecasting
- Real-time computation using posterior samples
- Visual comparison with confidence intervals
- Probability distributions showing likelihood

### LUCR Analysis Tab
- Baseline vs adjusted LUCR curves
- Confidence bands from Bayesian posterior
- Only affected by β multiplier (algorithmic efficiency)

### Data Tables Tab
- Browse and download all underlying data
- Export tables as CSV

**Example scenarios:**
- AGI Race (3-month compute doubling)
- Regulatory Slowdown (12-month doubling)
- Algorithmic Breakthrough (+20% translation efficiency)
- Sustained Efficiency Gains (1.2× β multiplier)

---

## 📚 Data Sources

- **METR Task Standard**: https://metr.github.io/autonomy-evals-guide/
- **Epoch AI**: https://epochai.org/
- **Epoch Capabilities Index**: Heim et al., 2024

---

## 🤝 Contributing

This project was developed for the Apart Research AGI forecasting competition. The analysis pipeline is designed to be:
- **Reproducible**: All data and scripts included
- **Transparent**: Clear methodology and assumptions
- **Interactive**: Streamlit app for exploring scenarios
- **Robust**: Bayesian uncertainty quantification throughout

---

## 📄 License

Data sources retain their original licenses. Analysis code provided for research purposes.

---

## 🙏 Acknowledgments

- METR team for benchmark data and evaluation framework
- Epoch AI for comprehensive AI training compute database
- Apart Research for organizing the forecasting competition

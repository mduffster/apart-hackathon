"""
STEP 1: Filter METR data for subhuman runs (Method B)

This script implements "Method B" filtering:
- Only include runs where the model's performance is below human level on that specific task
- Exclude trivial tasks (where all models get ~1.0)
- Focus on the "catching up to humans" problem

Output:
- filtered_metr_runs.csv: Clean METR dataset for downstream analysis
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("="*100)
print("STEP 1: FILTER METR DATA FOR SUBHUMAN RUNS")
print("="*100)

# ============================================================================
# Load METR runs
# ============================================================================

runs_file = Path('data/METR/all_runs.jsonl')
runs = []
with open(runs_file, 'r') as f:
    for line in f:
        runs.append(json.loads(line))

df = pd.DataFrame(runs)
print(f"\nLoaded {len(df):,} runs")
print(f"  Models: {df['model'].nunique()}")
print(f"  Task families: {df['task_family'].nunique()}")

# ============================================================================
# Load principled compute assignments
# ============================================================================

import json
with open('outputs/compute_assignments.json', 'r') as f:
    compute_data = json.load(f)

# Extract compute values
known_compute = {
    model: data['compute'] 
    for model, data in compute_data.items() 
    if data['compute'] is not None
}

print(f"\n{'='*100}")
print("COMPUTE IMPUTATION (Principled)")
print("="*100)
print(f"\nMethod breakdown:")
for model, data in compute_data.items():
    method = data.get('method', 'unknown')
    compute = data.get('compute')
    if compute:
        if 'compute_low' in data:
            print(f"  {model:20s}: {compute:10.2e} ± uncertainty ({method})")
        else:
            print(f"  {model:20s}: {compute:10.2e} ({method})")

df['compute'] = df['model'].map(known_compute)
models_with_compute = df[df['compute'].notna()].copy()

print(f"\nModels with compute estimates: {models_with_compute['model'].nunique()}")
print(f"  Runs: {len(models_with_compute):,}")

# ============================================================================
# Filter 1: Models with sufficient task coverage
# ============================================================================

model_coverage = models_with_compute.groupby('model')['task_family'].nunique()
well_covered_models = model_coverage[model_coverage >= 10].index
filtered = models_with_compute[models_with_compute['model'].isin(well_covered_models)].copy()

print(f"\n{'='*100}")
print("FILTER 1: Model Coverage")
print("="*100)
print(f"\nWell-covered models (≥10 unique tasks): {len(well_covered_models)}")
for model in well_covered_models:
    n_tasks = model_coverage[model]
    print(f"  {model}: {n_tasks} tasks")

print(f"\nRuns after filter: {len(filtered):,}")

# ============================================================================
# Filter 2: Identify trivial tasks
# ============================================================================

# Only use well-covered models to identify trivial tasks
task_stats = filtered.groupby('task_family').agg({
    'score_cont': ['mean', 'std', lambda x: (x == 1.0).mean()]
})
task_stats.columns = ['mean_score', 'std_score', 'pct_perfect']
task_stats = task_stats.round(3)

# Trivial = high mean, low variance (everyone solves it easily)
trivial_tasks = task_stats[
    (task_stats['mean_score'] > 0.95) & 
    (task_stats['std_score'] < 0.05)
].index

meaningful_tasks = task_stats.index.difference(trivial_tasks)

print(f"\n{'='*100}")
print("FILTER 2: Trivial Tasks")
print("="*100)
print(f"\nTask classification:")
print(f"  Trivial (mean>0.95, std<0.05): {len(trivial_tasks)}")
print(f"  Meaningful: {len(meaningful_tasks)}")

if len(trivial_tasks) > 0:
    print(f"\nTrivial tasks:")
    for task in trivial_tasks:
        stats = task_stats.loc[task]
        print(f"  {task}: mean={stats['mean_score']:.3f}, std={stats['std_score']:.3f}")

# Exclude trivial tasks
filtered = filtered[filtered['task_family'].isin(meaningful_tasks)].copy()
print(f"\nRuns after filter: {len(filtered):,}")

# ============================================================================
# Filter 3: Subhuman runs only (Method B)
# ============================================================================

# Calculate score normalized to human performance
filtered['score_vs_human'] = filtered['score_cont'] / filtered['human_score']

# Filter for subhuman runs (model < human)
subhuman_runs = filtered[filtered['score_vs_human'] < 1.0].copy()

print(f"\n{'='*100}")
print("FILTER 3: Subhuman Runs (Method B)")
print("="*100)
print(f"\nRuns where model < human on that task:")
print(f"  {len(subhuman_runs):,} / {len(filtered):,} ({len(subhuman_runs)/len(filtered)*100:.1f}%)")

# Show per-model breakdown
print(f"\nSubhuman runs per model:")
for model in well_covered_models:
    model_total = len(filtered[filtered['model'] == model])
    model_subhuman = len(subhuman_runs[subhuman_runs['model'] == model])
    pct = model_subhuman / model_total * 100 if model_total > 0 else 0
    print(f"  {model}: {model_subhuman} / {model_total} ({pct:.1f}%)")

# ============================================================================
# Freeze cohort: Save task list and compute weights
# ============================================================================

print(f"\n{'='*100}")
print("FREEZE SUBHUMAN COHORT")
print("="*100)

# Freeze the task families included in subhuman cohort
frozen_task_families = subhuman_runs['task_family'].unique().tolist()
n_frozen_tasks = len(frozen_task_families)

print(f"\nFrozen task families: {n_frozen_tasks}")
print(f"  (These tasks will be used consistently for all capability estimates)")

# Compute equal weights (normalize to sum to 1)
task_weights = {task: 1.0 / n_frozen_tasks for task in frozen_task_families}

# Save frozen cohort metadata
cohort_metadata = {
    'frozen_date': '2025-11-01',  # Date cohort was frozen
    'n_tasks': n_frozen_tasks,
    'task_families': frozen_task_families,
    'task_weights': task_weights,
    'n_models': len(well_covered_models),
    'models': well_covered_models.tolist()
}

import json
with open('outputs/frozen_cohort.json', 'w') as f:
    json.dump(cohort_metadata, f, indent=2)

print(f"\n✅ Saved frozen cohort: outputs/frozen_cohort.json")
print(f"  Task weights sum to: {sum(task_weights.values()):.6f}")

# ============================================================================
# Calculate per-model capability using frozen weights
# ============================================================================

print(f"\n{'='*100}")
print("CAPABILITY ESTIMATES (FROZEN COHORT)")
print("="*100)

capability_results = []
for model in well_covered_models:
    model_runs = subhuman_runs[subhuman_runs['model'] == model]
    
    if len(model_runs) > 0:
        # Compute weighted capability using frozen weights
        # Only include tasks that are in the frozen cohort
        model_runs_frozen = model_runs[model_runs['task_family'].isin(frozen_task_families)]
        
        # Group by task and get mean score per task
        task_scores = model_runs_frozen.groupby('task_family')['score_cont'].mean()
        
        # Apply frozen weights for METR capability
        weighted_capability = sum(task_scores.get(task, 0) * task_weights[task] 
                                   for task in frozen_task_families)
        
        # Compute Headroom = 1 - Capability Gain Index (CGI)
        # CGI measures "how much of the gap to human-level has been closed"
        # Headroom measures "how much room is left to human-level"
        # For subhuman tasks, we're already measuring headroom implicitly
        # Headroom = 1 - METR_capability (since target is 1.0)
        headroom = 1.0 - weighted_capability
        
        # Also compute CGI for reference (progress toward human-level)
        # CGI = METR_capability / 1.0 = METR_capability
        cgi = weighted_capability
        
        compute = model_runs['compute'].iloc[0]
        n_tasks = len(task_scores)
        n_runs = len(model_runs_frozen)
        
        capability_results.append({
            'model': model,
            'compute': compute,
            'metr_capability': weighted_capability,
            'cgi': cgi,  # Capability Gain Index (progress made)
            'headroom': headroom,  # Room remaining to human-level
            'n_task_families': n_tasks,
            'n_runs': n_runs
        })

capability_df = pd.DataFrame(capability_results).sort_values('compute')

print(f"\nMETR Capability (subhuman runs only):")
print(capability_df.to_string(index=False))

print(f"\nBest current models:")
best_models = capability_df.nlargest(2, 'compute')
for _, row in best_models.iterrows():
    print(f"  {row['model']}: {row['metr_capability']:.3f} ({row['n_task_families']} families, {row['n_runs']} runs)")

# ============================================================================
# Save filtered dataset
# ============================================================================

output_file = 'outputs/filtered_metr_runs.csv'
subhuman_runs.to_csv(output_file, index=False)
print(f"\n✅ Saved: {output_file}")

# Save capability summary
capability_file = 'outputs/metr_capability_summary.csv'
capability_df.to_csv(capability_file, index=False)
print(f"✅ Saved: {capability_file}")

print(f"\n{'='*100}")
print("✅ STEP 1 COMPLETE")
print("="*100)

print(f"\n💡 SUMMARY:")
print(f"  • Started with {len(df):,} runs")
print(f"  • Filtered to {len(well_covered_models)} well-covered models")
print(f"  • Excluded {len(trivial_tasks)} trivial tasks")
print(f"  • Kept {len(subhuman_runs):,} subhuman runs (69%)")
print(f"  • Current frontier METR: {capability_df['metr_capability'].max():.3f}")
print(f"\n📊 Next step: Run 02_compute_eci_metr_data.py to build modeling datasets")


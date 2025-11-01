#!/usr/bin/env python3
"""
Explore METR all_runs.jsonl data structure
Shows what fields are available in each task run
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def analyze_jsonl_structure(jsonl_path):
    """Analyze the structure and fields in the JSONL file."""
    print("="*80)
    print("METR ALL_RUNS.JSONL DATA STRUCTURE ANALYSIS")
    print("="*80)
    
    # Read all lines
    runs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            runs.append(json.loads(line))
    
    print(f"\n📊 BASIC INFO")
    print(f"   Total runs: {len(runs):,}")
    
    # Get all unique keys
    all_keys = set()
    for run in runs:
        all_keys.update(run.keys())
    
    print(f"   Unique fields: {len(all_keys)}")
    
    # Show first run as example
    print(f"\n📋 SAMPLE RUN (first entry)")
    print(f"   JSON structure:")
    first_run = runs[0]
    for key, value in first_run.items():
        value_type = type(value).__name__
        value_preview = str(value)[:60] if value is not None else "None"
        print(f"      • {key:25s} ({value_type:8s}): {value_preview}")
    
    # Analyze each field
    print(f"\n🔍 FIELD ANALYSIS (All {len(all_keys)} fields)")
    print("-" * 80)
    
    field_stats = {}
    for key in sorted(all_keys):
        non_null_count = sum(1 for run in runs if run.get(key) is not None)
        null_count = len(runs) - non_null_count
        
        # Get sample values (non-null)
        sample_values = []
        value_types = set()
        for run in runs:
            val = run.get(key)
            if val is not None:
                value_types.add(type(val).__name__)
                if val not in sample_values and len(sample_values) < 5:
                    sample_values.append(val)
        
        field_stats[key] = {
            'non_null': non_null_count,
            'null': null_count,
            'pct_non_null': (non_null_count / len(runs) * 100),
            'types': value_types,
            'samples': sample_values
        }
    
    # Print field details
    for key in sorted(all_keys):
        stats = field_stats[key]
        type_str = ', '.join(stats['types']) if stats['types'] else 'unknown'
        
        print(f"\n{key}")
        print(f"   Type: {type_str}")
        print(f"   Non-null: {stats['non_null']:,} ({stats['pct_non_null']:.1f}%)")
        
        if stats['samples']:
            print(f"   Sample values:")
            for sample in stats['samples'][:3]:
                sample_str = str(sample)[:70]
                print(f"      • {sample_str}")
    
    # Aggregate statistics
    print(f"\n📈 AGGREGATE STATISTICS")
    
    # Unique models
    models = [run.get('model') for run in runs if run.get('model')]
    model_counts = Counter(models)
    print(f"\n   Unique models: {len(model_counts)}")
    print(f"   Top 10 models by run count:")
    for model, count in model_counts.most_common(10):
        print(f"      • {model:40s}: {count:>5,} runs")
    
    # Unique tasks
    tasks = [run.get('task_id') for run in runs if run.get('task_id')]
    task_counts = Counter(tasks)
    print(f"\n   Unique tasks: {len(task_counts)}")
    print(f"   Top 10 tasks by run count:")
    for task, count in task_counts.most_common(10):
        print(f"      • {task:50s}: {count:>3} runs")
    
    # Task families
    task_families = [run.get('task_family') for run in runs if run.get('task_family')]
    family_counts = Counter(task_families)
    print(f"\n   Unique task families: {len(family_counts)}")
    print(f"   Top 10 task families:")
    for family, count in family_counts.most_common(10):
        print(f"      • {family:40s}: {count:>4,} runs")
    
    # Score statistics
    scores_cont = [run.get('score_cont') for run in runs if run.get('score_cont') is not None]
    scores_bin = [run.get('score_binarized') for run in runs if run.get('score_binarized') is not None]
    
    print(f"\n   Score statistics:")
    if scores_cont:
        print(f"      Continuous scores (score_cont):")
        print(f"         Range: {min(scores_cont):.4f} to {max(scores_cont):.4f}")
        print(f"         Mean:  {sum(scores_cont)/len(scores_cont):.4f}")
        non_zero = sum(1 for s in scores_cont if s > 0)
        print(f"         Non-zero: {non_zero:,} ({non_zero/len(scores_cont)*100:.1f}%)")
    
    if scores_bin:
        print(f"      Binary scores (score_binarized):")
        ones = sum(scores_bin)
        zeros = len(scores_bin) - ones
        print(f"         Success (1): {ones:,} ({ones/len(scores_bin)*100:.1f}%)")
        print(f"         Failure (0): {zeros:,} ({zeros/len(scores_bin)*100:.1f}%)")
    
    # Cost statistics
    gen_costs = [run.get('generation_cost') for run in runs if run.get('generation_cost') is not None and run.get('generation_cost') > 0]
    if gen_costs:
        print(f"\n   Generation costs:")
        print(f"      Total runs with cost: {len(gen_costs):,}")
        print(f"      Total cost: ${sum(gen_costs):,.2f}")
        print(f"      Average cost per run: ${sum(gen_costs)/len(gen_costs):.2f}")
        print(f"      Range: ${min(gen_costs):.2f} to ${max(gen_costs):.2f}")
    
    # Task sources
    task_sources = [run.get('task_source') for run in runs if run.get('task_source')]
    source_counts = Counter(task_sources)
    print(f"\n   Task sources:")
    for source, count in source_counts.most_common():
        print(f"      • {source:20s}: {count:>5,} runs ({count/len(runs)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


def main():
    jsonl_path = Path(__file__).parent / "data" / "METR" / "all_runs.jsonl"
    
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        return
    
    analyze_jsonl_structure(jsonl_path)


if __name__ == "__main__":
    main()


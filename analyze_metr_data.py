#!/usr/bin/env python3
"""
METR Benchmark Data Analysis Script
Converts YAML benchmark data to tabular format and finds overlap with AI models datasets
"""

import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
import re


def load_metr_yaml(yaml_path):
    """Load and parse the METR benchmark YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def yaml_to_dataframe(metr_data):
    """Convert METR YAML data to a flat pandas DataFrame."""
    rows = []
    
    benchmark_name = metr_data.get('benchmark_name', 'Unknown')
    results = metr_data.get('results', {})
    
    for model_name, model_data in results.items():
        agents = model_data.get('agents', {})
        release_date = model_data.get('release_date', None)
        
        for agent_name, metrics in agents.items():
            row = {
                'model_name': model_name,
                'benchmark': benchmark_name,
                'agent': agent_name,
                'release_date': release_date,
                'is_sota': model_data.get('is_sota', False),
            }
            
            # Add p50 horizon length metrics
            if 'p50_horizon_length' in metrics:
                row['p50_estimate'] = metrics['p50_horizon_length'].get('estimate')
                row['p50_ci_low'] = metrics['p50_horizon_length'].get('ci_low')
                row['p50_ci_high'] = metrics['p50_horizon_length'].get('ci_high')
            
            # Add p80 horizon length metrics
            if 'p80_horizon_length' in metrics:
                row['p80_estimate'] = metrics['p80_horizon_length'].get('estimate')
                row['p80_ci_low'] = metrics['p80_horizon_length'].get('ci_low')
                row['p80_ci_high'] = metrics['p80_horizon_length'].get('ci_high')
            
            # Add average score
            if 'average_score' in metrics:
                row['average_score'] = metrics['average_score'].get('estimate')
            
            # Add usage data if available
            if 'usage' in metrics:
                row['working_time'] = metrics['usage'].get('working_time')
                row['cost_usd'] = metrics['usage'].get('usd')
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def normalize_model_name(name):
    """Normalize model names for better matching."""
    if pd.isna(name):
        return ""
    
    name = str(name).lower()
    # Remove version suffixes like _20241022
    name = re.sub(r'_\d{8}', '', name)
    # Remove version numbers with underscores
    name = re.sub(r'_v?\d+(_\d+)*$', '', name)
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    # Remove extra spaces
    name = ' '.join(name.split())
    
    return name


def find_model_overlap(metr_df, ai_models_df, dataset_name):
    """Find overlapping models between METR data and AI models dataset."""
    # Normalize model names
    metr_models = set(metr_df['model_name'].apply(normalize_model_name))
    ai_models = set(ai_models_df['Model'].apply(normalize_model_name))
    
    # Direct matches
    direct_matches = metr_models & ai_models
    
    # Fuzzy matches (contains)
    fuzzy_matches = set()
    for metr_model in metr_models:
        for ai_model in ai_models:
            # Check if one is contained in the other
            if metr_model in ai_model or ai_model in metr_model:
                if len(metr_model) > 3 and len(ai_model) > 3:  # Avoid short false matches
                    fuzzy_matches.add((metr_model, ai_model))
    
    return {
        'dataset': dataset_name,
        'total_metr_models': len(metr_models),
        'total_ai_models': len(ai_models),
        'direct_matches': len(direct_matches),
        'fuzzy_matches': len(fuzzy_matches),
        'direct_match_list': direct_matches,
        'fuzzy_match_list': fuzzy_matches
    }


def main():
    print("="*80)
    print("METR BENCHMARK DATA ANALYSIS")
    print("="*80)
    
    # Load METR YAML data
    metr_path = Path(__file__).parent / "data" / "METR" / "benchmark_results.yaml"
    print(f"\n📂 Loading METR data from: {metr_path}")
    
    metr_data = load_metr_yaml(metr_path)
    print(f"   Benchmark: {metr_data.get('benchmark_name', 'Unknown')}")
    print(f"   Total models: {len(metr_data.get('results', {}))}")
    
    # Convert to DataFrame
    print("\n🔄 Converting YAML to tabular format...")
    metr_df = yaml_to_dataframe(metr_data)
    
    print(f"   Rows created: {len(metr_df)}")
    print(f"   Columns: {len(metr_df.columns)}")
    
    # Display summary statistics
    print("\n📊 METR DATA SUMMARY")
    print(f"   Unique models: {metr_df['model_name'].nunique()}")
    print(f"   Unique agents: {metr_df['agent'].nunique()}")
    print(f"   SOTA models: {metr_df['is_sota'].sum()}")
    print(f"   Date range: {metr_df['release_date'].min()} to {metr_df['release_date'].max()}")
    
    print("\n📈 PERFORMANCE METRICS")
    print(f"   Average score range: {metr_df['average_score'].min():.3f} to {metr_df['average_score'].max():.3f}")
    print(f"   Average score mean: {metr_df['average_score'].mean():.3f}")
    print(f"   P50 estimate range: {metr_df['p50_estimate'].min():.3f} to {metr_df['p50_estimate'].max():.3f}")
    
    # Show top performers
    print("\n🏆 TOP 10 MODELS BY AVERAGE SCORE")
    top_models = metr_df.nlargest(10, 'average_score')[['model_name', 'release_date', 'average_score', 'is_sota']]
    for idx, row in top_models.iterrows():
        sota_marker = "⭐" if row['is_sota'] else "  "
        print(f"   {sota_marker} {row['model_name']:35s} - Score: {row['average_score']:.3f} - Released: {row['release_date']}")
    
    # Display sample data
    print("\n👀 SAMPLE DATA (first 5 rows)")
    print("\n" + metr_df.head(5).to_string(index=False, max_colwidth=30))
    
    # Save to CSV
    output_path = Path(__file__).parent / "data" / "METR" / "benchmark_results.csv"
    metr_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved tabular data to: {output_path}")
    
    # Load AI models datasets
    print("\n" + "="*80)
    print("FINDING OVERLAP WITH AI MODELS DATASETS")
    print("="*80)
    
    data_dir = Path(__file__).parent / "data" / "ai_models"
    datasets = {
        'all_ai_models': pd.read_csv(data_dir / "all_ai_models.csv", low_memory=False),
        'notable_ai_models': pd.read_csv(data_dir / "notable_ai_models.csv", low_memory=False),
        'large_scale_ai_models': pd.read_csv(data_dir / "large_scale_ai_models.csv", low_memory=False),
        'frontier_ai_models': pd.read_csv(data_dir / "frontier_ai_models.csv", low_memory=False)
    }
    
    print("\n🔍 Analyzing model name overlap...")
    
    overlap_results = []
    for dataset_name, df in datasets.items():
        result = find_model_overlap(metr_df, df, dataset_name)
        overlap_results.append(result)
    
    # Display overlap results
    print("\n📊 OVERLAP SUMMARY")
    print(f"\n{'Dataset':<25} {'AI Models':<12} {'Direct Matches':<15} {'Fuzzy Matches':<15} {'Match %':<10}")
    print("-" * 80)
    
    for result in overlap_results:
        match_pct = (result['direct_matches'] + result['fuzzy_matches']) / result['total_metr_models'] * 100
        print(f"{result['dataset']:<25} {result['total_ai_models']:<12} {result['direct_matches']:<15} {result['fuzzy_matches']:<15} {match_pct:>8.1f}%")
    
    # Find best dataset
    best_dataset = max(overlap_results, key=lambda x: x['direct_matches'] + x['fuzzy_matches'])
    
    print("\n" + "="*80)
    print(f"🏆 BEST OVERLAP: {best_dataset['dataset']}")
    print("="*80)
    print(f"   Total matches: {best_dataset['direct_matches'] + best_dataset['fuzzy_matches']}")
    print(f"   Direct matches: {best_dataset['direct_matches']}")
    print(f"   Fuzzy matches: {best_dataset['fuzzy_matches']}")
    
    if best_dataset['direct_matches'] > 0:
        print("\n   Direct matches found:")
        for match in sorted(list(best_dataset['direct_match_list']))[:10]:
            print(f"      • {match}")
        if len(best_dataset['direct_match_list']) > 10:
            print(f"      ... and {len(best_dataset['direct_match_list']) - 10} more")
    
    if best_dataset['fuzzy_matches'] > 0:
        print("\n   Fuzzy matches found (METR model → AI model):")
        for metr_model, ai_model in sorted(list(best_dataset['fuzzy_match_list']))[:15]:
            print(f"      • '{metr_model}' ↔ '{ai_model}'")
        if len(best_dataset['fuzzy_match_list']) > 15:
            print(f"      ... and {len(best_dataset['fuzzy_match_list']) - 15} more")
    
    # Show METR models not found in any dataset
    print("\n❓ METR MODELS NOT FOUND IN AI DATASETS")
    all_ai_models_normalized = set()
    for df in datasets.values():
        all_ai_models_normalized.update(df['Model'].apply(normalize_model_name))
    
    metr_models_normalized = set(metr_df['model_name'].apply(normalize_model_name))
    not_found = []
    for metr_model in metr_models_normalized:
        found = False
        for ai_model in all_ai_models_normalized:
            if metr_model in ai_model or ai_model in metr_model:
                if len(metr_model) > 3 and len(ai_model) > 3:
                    found = True
                    break
        if not found:
            # Get original name
            original = metr_df[metr_df['model_name'].apply(normalize_model_name) == metr_model]['model_name'].iloc[0]
            not_found.append(original)
    
    if not_found:
        print(f"\n   {len(not_found)} METR models not found:")
        for model in sorted(set(not_found))[:15]:
            print(f"      • {model}")
        if len(not_found) > 15:
            print(f"      ... and {len(not_found) - 15} more")
    else:
        print("   ✓ All METR models found in AI datasets!")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  • CSV file: {output_path}")
    print(f"  • Best dataset for joining: {best_dataset['dataset']}")


if __name__ == "__main__":
    main()



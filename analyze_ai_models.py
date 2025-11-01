#!/usr/bin/env python3
"""
AI Models Data Analysis Script
Performs descriptive analysis on CSV files in data/ai_models/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_csv_files(data_dir):
    """Load all CSV files from the specified directory."""
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return {}
    
    datasets = {}
    for csv_file in csv_files:
        print(f"\nLoading {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            datasets[csv_file.stem] = df
            print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"  ✗ Error loading {csv_file.name}: {e}")
    
    return datasets


def analyze_dataset(name, df):
    """Perform comprehensive descriptive analysis on a dataset."""
    print("\n" + "="*80)
    print(f"DATASET: {name}")
    print("="*80)
    
    # Basic information
    print(f"\n📊 BASIC INFORMATION")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns):,}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    print(f"\n📋 COLUMN OVERVIEW")
    print(f"   Total columns: {len(df.columns)}")
    
    # Data types
    print(f"\n🔤 DATA TYPES")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    # Missing data analysis
    print(f"\n❓ MISSING DATA ANALYSIS")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(f"   Columns with missing data: {len(missing_df)}/{len(df.columns)}")
        print(f"\n   Top 10 columns with most missing data:")
        for idx, (col, row) in enumerate(missing_df.head(10).iterrows(), 1):
            print(f"   {idx:2d}. {col[:50]:50s} - {row['Missing Count']:>7,} ({row['Percentage']:>6.2f}%)")
    else:
        print("   ✓ No missing data found!")
    
    # Numerical columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 NUMERICAL COLUMNS ({len(numeric_cols)} columns)")
        print("\n   Summary Statistics:")
        desc = df[numeric_cols].describe()
        # Show first few numerical columns
        for col in numeric_cols[:5]:
            print(f"\n   {col}:")
            print(f"      Count: {desc[col]['count']:,.0f}")
            print(f"      Mean:  {desc[col]['mean']:,.2f}")
            print(f"      Std:   {desc[col]['std']:,.2f}")
            print(f"      Min:   {desc[col]['min']:,.2f}")
            print(f"      25%:   {desc[col]['25%']:,.2f}")
            print(f"      50%:   {desc[col]['50%']:,.2f}")
            print(f"      75%:   {desc[col]['75%']:,.2f}")
            print(f"      Max:   {desc[col]['max']:,.2f}")
        
        if len(numeric_cols) > 5:
            print(f"\n   ... and {len(numeric_cols) - 5} more numerical columns")
    
    # Categorical/text columns analysis
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        print(f"\n📝 TEXT/CATEGORICAL COLUMNS ({len(text_cols)} columns)")
        print("\n   Columns with unique value counts:")
        
        # Sort by number of unique values
        unique_counts = []
        for col in text_cols:
            unique_counts.append((col, df[col].nunique()))
        unique_counts.sort(key=lambda x: x[1])
        
        # Show first 10
        for col, unique_count in unique_counts[:10]:
            total_non_null = df[col].notna().sum()
            print(f"   {col[:50]:50s} - {unique_count:>6,} unique values ({total_non_null:>7,} non-null)")
            
            # Show top values for columns with reasonable number of unique values
            if unique_count <= 20 and unique_count > 0:
                top_values = df[col].value_counts().head(5)
                for val, count in top_values.items():
                    val_str = str(val)[:40]
                    print(f"      • {val_str:40s}: {count:>6,}")
        
        if len(text_cols) > 10:
            print(f"\n   ... and {len(text_cols) - 10} more text columns")
    
    # Sample rows
    print(f"\n👀 SAMPLE DATA (first 3 rows)")
    print("\n" + df.head(3).to_string(max_colwidth=50))
    
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_cols': len(numeric_cols),
        'text_cols': len(text_cols),
        'missing_cols': len(missing_df)
    }


def summarize_all_datasets(datasets):
    """Create a summary comparison of all datasets."""
    print("\n" + "="*80)
    print("SUMMARY: ALL DATASETS")
    print("="*80)
    
    summary_data = []
    for name, df in datasets.items():
        summary_data.append({
            'Dataset': name,
            'Rows': f"{len(df):,}",
            'Columns': len(df.columns),
            'Numeric Cols': len(df.select_dtypes(include=[np.number]).columns),
            'Text Cols': len(df.select_dtypes(include=['object']).columns),
            'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Column overlap analysis
    if len(datasets) > 1:
        print(f"\n📊 COLUMN OVERLAP ANALYSIS")
        all_columns = {}
        for name, df in datasets.items():
            all_columns[name] = set(df.columns)
        
        # Find common columns
        common_cols = set.intersection(*all_columns.values())
        print(f"\n   Columns present in ALL datasets: {len(common_cols)}")
        if len(common_cols) <= 20:
            print(f"   {', '.join(sorted(list(common_cols))[:10])}")
            if len(common_cols) > 10:
                print(f"   ... and {len(common_cols) - 10} more")


def main():
    """Main execution function."""
    print("="*80)
    print("AI MODELS DATA ANALYSIS")
    print("="*80)
    
    # Define data directory
    data_dir = Path(__file__).parent / "data" / "ai_models"
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist!")
        sys.exit(1)
    
    # Load all CSV files
    print(f"\nScanning directory: {data_dir}")
    datasets = load_csv_files(data_dir)
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        sys.exit(1)
    
    print(f"\n✓ Successfully loaded {len(datasets)} dataset(s)")
    
    # Analyze each dataset
    results = {}
    for name, df in datasets.items():
        results[name] = analyze_dataset(name, df)
    
    # Create overall summary
    summarize_all_datasets(datasets)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nFor detailed analysis, you can:")
    print("  • Import this data into Jupyter notebooks")
    print("  • Use pandas/matplotlib for visualizations")
    print("  • Filter and query specific columns")
    print("  • Export subsets for further analysis")


if __name__ == "__main__":
    main()


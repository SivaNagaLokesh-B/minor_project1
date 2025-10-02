#!/usr/bin/env python3
"""
Quick script to inspect the current dataset
"""

import pandas as pd
import os
from pathlib import Path

def inspect_current_data():
    data_paths = {
        'sensor_data': 'data/raw/sensor_data.csv',
        'maintenance_logs': 'data/raw/maintenance_logs.csv', 
        'operational_data': 'data/raw/operational_data.csv'
    }
    
    print("🔍 Inspecting Current Dataset\n")
    print("="*50)
    
    for name, path in data_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"\n📊 {name.upper().replace('_', ' ')}:")
            print(f"   📁 File: {path}")
            print(f"   📈 Records: {len(df):,}")
            print(f"   🏭 Equipment IDs: {df['equipment_id'].nunique() if 'equipment_id' in df.columns else 'N/A'}")
            print(f"   📅 Date range: {df['timestamp'].min() if 'timestamp' in df.columns else 'N/A'} to {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}")
            print(f"   🔧 Columns: {list(df.columns)}")
            
            # Show sample data
            print(f"   👀 Sample data:")
            print(df.head(2).to_string(index=False))
        else:
            print(f"\n❌ {name}: File not found at {path}")
        
        print("-"*50)

if __name__ == "__main__":
    inspect_current_data()
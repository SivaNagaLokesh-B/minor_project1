#!/usr/bin/env python3
"""
Test the fixed pipeline
"""

import logging
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.real_dataset_loader import RealDatasetLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dataset_loading():
    """Test if the dataset loader works correctly"""
    try:
        loader = RealDatasetLoader()
        
        print("🧪 Testing dataset loading...")
        
        # Test AI4I dataset
        dataset = loader.load_dataset('ai4i2020', download=True)
        
        print("✅ Dataset loaded successfully!")
        print(f"📊 Sensor records: {len(dataset['sensor'])}")
        print(f"🔧 Maintenance records: {len(dataset['maintenance'])}")
        print(f"🏭 Operational records: {len(dataset['operational'])}")
        
        # Check timestamp validity
        sensor_data = dataset['sensor']
        invalid_timestamps = sensor_data['timestamp'].isna().sum()
        print(f"⏰ Invalid timestamps: {invalid_timestamps}")
        
        if invalid_timestamps == 0:
            print("🎉 All timestamps are valid!")
        else:
            print(f"⚠️  Found {invalid_timestamps} invalid timestamps")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("\n🎯 Now run: python run_real_data_pipeline.py --dataset ai4i2020")
    else:
        print("\n💥 Fix the issues before running the pipeline")
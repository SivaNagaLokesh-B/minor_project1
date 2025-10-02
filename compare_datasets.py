#!/usr/bin/env python3
"""
Compare different datasets for predictive maintenance
"""

import logging
from src.data_processing.dataset_loader import DatasetLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_datasets():
    """Compare available datasets and their characteristics"""
    
    loader = DatasetLoader()
    datasets_info = {
        'nasa_cmapss': {
            'description': 'NASA Turbofan Engine Degradation Data',
            'equipment_type': 'Aircraft Engines',
            'sensors': '21 sensor readings',
            'samples': '~20,000+ per subset',
            'failure_modes': 'Engine degradation',
            'best_for': 'RUL prediction, time-series analysis'
        },
        'ai4i2020': {
            'description': 'AI4I 2020 Predictive Maintenance Dataset',
            'equipment_type': 'Industrial Machines',
            'sensors': 'Temperature, torque, rotation',
            'samples': '10,000 records',
            'failure_modes': 'Tool wear, failures',
            'best_for': 'Classification, failure prediction'
        },
        'bearing': {
            'description': 'Bearing Vibration Data',
            'equipment_type': 'Rotating Machinery',
            'sensors': 'Vibration, acceleration',
            'samples': 'Varies by dataset',
            'failure_modes': 'Bearing degradation',
            'best_for': 'Vibration analysis, early detection'
        }
    }
    
    print("🔍 Available Predictive Maintenance Datasets")
    print("=" * 80)
    
    for dataset_name, info in datasets_info.items():
        print(f"\n📊 Dataset: {dataset_name.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Equipment: {info['equipment_type']}")
        print(f"   Sensors: {info['sensors']}")
        print(f"   Samples: {info['samples']}")
        print(f"   Failure Modes: {info['failure_modes']}")
        print(f"   Best For: {info['best_for']}")
        
        # Try to load dataset info
        try:
            data = loader.load_dataset(dataset_name, download=False)
            if 'sensor' in data:
                print(f"   ✅ Available: {len(data['sensor'])} sensor records")
            if 'maintenance' in data:
                print(f"   ✅ Maintenance events: {len(data['maintenance'])}")
        except Exception as e:
            print(f"   ❌ Load error: {e}")
    
    print("\n🎯 Recommendation:")
    print("   - Start with 'ai4i2020' for quick results")
    print("   - Use 'nasa_cmapss' for advanced RUL prediction")
    print("   - Try 'bearing' for vibration analysis")

if __name__ == "__main__":
    compare_datasets()
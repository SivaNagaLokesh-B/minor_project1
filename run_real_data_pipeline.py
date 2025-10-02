#!/usr/bin/env python3
"""
Main pipeline for training with real predictive maintenance datasets - FIXED VERSION
"""

import logging
import argparse
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_processing.real_dataset_loader import RealDatasetLoader
    from src.data_processing.data_loader import DataManager
    from src.feature_engineering.feature_pipeline import FeatureEngineer
    from src.training.trainer import ModelTrainer
    from src.deployment.prediction_system import ProductionSystem
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all modules are in the correct location")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class RealDataPipeline:
    def __init__(self):
        self.dataset_loader = RealDatasetLoader()
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration"""
        return {
            'data_paths': {
                'raw_data': 'data/raw/sensor_data.csv',
                'maintenance_logs': 'data/raw/maintenance_logs.csv',
                'operational_data': 'data/raw/operational_data.csv'
            },
            'processed_paths': {
                'cleaned_data': 'data/processed/cleaned_sensor_data.csv',
                'feature_data': 'data/features/engineered_features.csv',
                'labels': 'data/processed/labels.csv'
            },
            'model_paths': {
                'saved_models': 'models/',
                'scalers': 'models/scalers/'
            }
        }
    
    def run_pipeline(self, dataset_name, download=True):
        """Run complete pipeline with real dataset"""
        logger.info(f"🚀 Starting Predictive Maintenance Pipeline with {dataset_name}")
        
        try:
            # Phase 1: Load Real Dataset
            logger.info("📊 Phase 1: Loading Real Dataset")
            dataset = self.dataset_loader.load_dataset(dataset_name, download=download)
            
            # Save dataset to standard paths
            self._save_dataset(dataset)
            
            # Phase 2: Data Processing
            logger.info("🔧 Phase 2: Data Processing")
            data_manager = DataManager(self.config)
            processed_data = data_manager.clean_and_preprocess(dataset)
            
            # Phase 3: Feature Engineering
            logger.info("⚡ Phase 3: Feature Engineering")
            feature_engineer = FeatureEngineer({})
            features, labels = feature_engineer.create_features_and_labels(processed_data)
            
            # Phase 4: Model Training
            logger.info("🤖 Phase 4: Model Training")
            trainer = ModelTrainer({})
            ensemble_model = trainer.train_ensemble(features, labels)
            
            # Phase 5: Validation
            logger.info("✅ Phase 5: Model Validation")
            validation_results = trainer.validate_model(ensemble_model, features, labels)
            
            # Phase 6: Deployment
            logger.info("🚀 Phase 6: Deployment Setup")
            production_system = ProductionSystem(ensemble_model, self.config, {})
            production_system.save_pipeline()
            
            logger.info(f"🎉 Pipeline completed successfully with {dataset_name}!")
            self._print_dataset_stats(dataset)
            
            return production_system
            
        except Exception as e:
            logger.error(f"❌ Error in pipeline: {e}")
            raise
    
    def _save_dataset(self, dataset):
        """Save dataset to standard file paths"""
        # Ensure timestamps are properly formatted before saving
        for key in ['sensor', 'maintenance', 'operational']:
            if key in dataset and 'timestamp' in dataset[key].columns:
                dataset[key]['timestamp'] = pd.to_datetime(dataset[key]['timestamp'], errors='coerce')
        
        dataset['sensor'].to_csv(self.config['data_paths']['raw_data'], index=False)
        dataset['maintenance'].to_csv(self.config['data_paths']['maintenance_logs'], index=False)
        dataset['operational'].to_csv(self.config['data_paths']['operational_data'], index=False)
        
        logger.info(f"💾 Dataset saved:")
        logger.info(f"   - Sensor data: {len(dataset['sensor'])} records")
        logger.info(f"   - Maintenance logs: {len(dataset['maintenance'])} records")
        logger.info(f"   - Operational data: {len(dataset['operational'])} records")
    
    def _print_dataset_stats(self, dataset):
        """Print dataset statistics"""
        sensor_data = dataset['sensor']
        
        logger.info("📈 Dataset Statistics:")
        logger.info(f"   Equipment count: {sensor_data['equipment_id'].nunique()}")
        logger.info(f"   Total records: {len(sensor_data)}")
        
        if 'failure_imminent' in sensor_data.columns:
            failure_rate = sensor_data['failure_imminent'].mean() * 100
            logger.info(f"   Failure rate: {failure_rate:.2f}%")
        
        if 'timestamp' in sensor_data.columns:
            logger.info(f"   Date range: {sensor_data['timestamp'].min()} to {sensor_data['timestamp'].max()}")
        
        # Sensor statistics
        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['degradation_level', 'failure_imminent']:
                logger.info(f"   {col}: mean={sensor_data[col].mean():.2f}, std={sensor_data[col].std():.2f}")

def list_datasets():
    """List available datasets"""
    loader = RealDatasetLoader()
    datasets = loader.list_datasets()
    
    print("\n🔍 Available Real Datasets:")
    print("=" * 50)
    for dataset in datasets:
        info = loader.get_dataset_info(dataset)
        print(f"\n📊 {dataset.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Features: {len(info['features'])} sensor parameters")
        print(f"   Target: {info['target']}")
        print(f"   URL: {info['url']}")

def main():
    parser = argparse.ArgumentParser(description='Train with Real Predictive Maintenance Datasets')
    parser.add_argument('--dataset', 
                       choices=['ai4i2020', 'nasa_turbofan', 'bearing_vibration'],
                       default='ai4i2020',
                       help='Dataset to use for training')
    parser.add_argument('--no-download', action='store_true',
                       help='Use local data instead of downloading')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    pipeline = RealDataPipeline()
    pipeline.run_pipeline(args.dataset, download=not args.no_download)

if __name__ == "__main__":
    main()
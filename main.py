#!/usr/bin/env python3
"""
Main execution script for Predictive Maintenance Project
"""

import warnings
warnings.filterwarnings('ignore')

import yaml
import argparse
import logging
from pathlib import Path
import os

# Set up logging FIRST, before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Now import other modules
from src.data_processing.data_loader import DataManager
try:
    from src.data_processing.generate_synthetic_data import generate_synthetic_data, generate_operational_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all required files exist in src/data_processing/")
    # Create fallback functions
    def generate_synthetic_data(*args, **kwargs):
        raise ImportError("generate_synthetic_data module not available")
    
    def generate_operational_data(*args, **kwargs):
        raise ImportError("generate_operational_data module not available")
from src.feature_engineering.feature_pipeline import FeatureEngineer
from src.modeling.ensemble_model import PredictiveMaintenanceEnsemble
from src.training.trainer import ModelTrainer
from src.deployment.prediction_system import ProductionSystem

class PredictiveMaintenancePipeline:
    def __init__(self, config_path="config/paths.yaml", model_config_path="config/model_params.yaml"):
        self.config = self.load_config(config_path)
        self.model_config = self.load_config(model_config_path)
        self.setup_directories()
        
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def setup_directories(self):
        """Create necessary directories"""
        for path_group in self.config.values():
            for path in path_group.values():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def check_and_generate_data(self):
        """Check if data exists, generate synthetic data if not"""
        data_paths = self.config['data_paths']
        
        if not all(os.path.exists(path) for path in data_paths.values()):
            logger.warning("Data files not found. Generating synthetic data...")
            self.generate_synthetic_data()
        else:
            logger.info("Data files found. Loading existing data...")
    
    def generate_synthetic_data(self):
        """Generate synthetic predictive maintenance data"""
        try:
            sensor_data, maintenance_data = generate_synthetic_data(num_equipment=10, days_of_data=90)
            operational_data = generate_operational_data(sensor_data)
            
            # Save generated data
            sensor_data.to_csv(self.config['data_paths']['raw_data'], index=False)
            maintenance_data.to_csv(self.config['data_paths']['maintenance_logs'], index=False)
            operational_data.to_csv(self.config['data_paths']['operational_data'], index=False)
            
            logger.info(f"Synthetic data generated: {len(sensor_data):,} sensor records")
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            raise
    
    def run_full_pipeline(self):
        """Execute complete pipeline from data to deployment"""
        logger.info("Starting Predictive Maintenance Pipeline")
        
        try:
            # Check and generate data if needed
            self.check_and_generate_data()
            
            # Phase 1: Data Processing
            logger.info("Phase 1: Data Processing")
            data_manager = DataManager(self.config)
            raw_data = data_manager.load_all_data()
            processed_data = data_manager.clean_and_preprocess(raw_data)
            
            # Phase 2: Feature Engineering
            logger.info("Phase 2: Feature Engineering")
            feature_engineer = FeatureEngineer(self.model_config)
            features, labels = feature_engineer.create_features_and_labels(processed_data)
            
            # Phase 3: Model Training
            logger.info("Phase 3: Model Training")
            trainer = ModelTrainer(self.model_config)
            ensemble_model = trainer.train_ensemble(features, labels)
            
            # Phase 4: Validation
            logger.info("Phase 4: Model Validation")
            validation_results = trainer.validate_model(ensemble_model, features, labels)
            
            # Phase 5: Deployment Ready
            logger.info("Phase 5: Deployment Setup")
            production_system = ProductionSystem(ensemble_model, self.config, self.model_config)
            production_system.save_pipeline()
            
            logger.info("Pipeline completed successfully!")
            return production_system
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Predictive Maintenance Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'serve'], default='train',
                       help='Operation mode: train, predict, or serve')
    parser.add_argument('--data', type=str, help='Path to data for prediction')
    parser.add_argument('--port', type=int, default=8000, help='Port for API server')
    
    args = parser.parse_args()
    
    pipeline = PredictiveMaintenancePipeline()
    
    if args.mode == 'train':
        pipeline.run_full_pipeline()
        
    elif args.mode == 'predict':
        if not args.data:
            raise ValueError("Please provide --data path for prediction")
        # Load new data and run inference
        # predictions = pipeline.run_inference(args.data)
        # print(predictions)
        
    elif args.mode == 'serve':
        # Start API server
        from src.deployment.api_server import start_server
        start_server(port=args.port)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Pipeline for training with real predictive maintenance datasets
"""

import logging
import argparse
from src.data_processing.dataset_loader import DatasetLoader
from src.data_processing.data_loader import DataManager
from src.feature_engineering.feature_pipeline import FeatureEngineer
from src.modeling.ensemble_model import PredictiveMaintenanceEnsemble
from src.training.trainer import ModelTrainer
from src.deployment.prediction_system import ProductionSystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDatasetPipeline:
    def __init__(self, config_path="config/paths.yaml", model_config_path="config/model_params.yaml"):
        self.config = self.load_config(config_path)
        self.model_config = self.load_config(model_config_path)
        self.dataset_loader = DatasetLoader()
        
    def load_config(self, config_path):
        import yaml
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def run_pipeline(self, dataset_name, **dataset_kwargs):
        """Run complete pipeline with specified dataset"""
        logger.info(f"Starting Pipeline with Dataset: {dataset_name}")
        
        try:
            # Phase 1: Load Real Dataset
            logger.info("Phase 1: Loading Real Dataset")
            dataset = self.dataset_loader.load_dataset(dataset_name, **dataset_kwargs)
            
            # Save dataset to expected paths
            self._save_dataset(dataset)
            
            # Phase 2: Data Processing
            logger.info("Phase 2: Data Processing")
            data_manager = DataManager(self.config)
            processed_data = data_manager.clean_and_preprocess(dataset)
            
            # Phase 3: Feature Engineering
            logger.info("Phase 3: Feature Engineering")
            feature_engineer = FeatureEngineer(self.model_config)
            features, labels = feature_engineer.create_features_and_labels(processed_data)
            
            # Phase 4: Model Training
            logger.info("Phase 4: Model Training")
            trainer = ModelTrainer(self.model_config)
            ensemble_model = trainer.train_ensemble(features, labels)
            
            # Phase 5: Validation
            logger.info("Phase 5: Model Validation")
            validation_results = trainer.validate_model(ensemble_model, features, labels)
            
            # Phase 6: Deployment Ready
            logger.info("Phase 6: Deployment Setup")
            production_system = ProductionSystem(ensemble_model, self.config, self.model_config)
            production_system.save_pipeline()
            
            logger.info(f"✅ Pipeline completed successfully with {dataset_name}!")
            self._print_results(validation_results)
            
            return production_system
            
        except Exception as e:
            logger.error(f"❌ Error in pipeline: {e}")
            raise
    
    def _save_dataset(self, dataset):
        """Save loaded dataset to expected file paths"""
        dataset['sensor'].to_csv(self.config['data_paths']['raw_data'], index=False)
        dataset['maintenance'].to_csv(self.config['data_paths']['maintenance_logs'], index=False)
        if 'operational' in dataset and not dataset['operational'].empty:
            dataset['operational'].to_csv(self.config['data_paths']['operational_data'], index=False)
    
    def _print_results(self, validation_results):
        """Print validation results"""
        if isinstance(validation_results, dict):
            logger.info("📊 Validation Results:")
            for metric, value in validation_results.items():
                logger.info(f"   {metric}: {value:.4f}")
        else:
            logger.info(f"📊 Validation Score: {validation_results:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train with Real Predictive Maintenance Datasets')
    parser.add_argument('--dataset', choices=['nasa_cmapss', 'ai4i2020', 'bearing', 'turbofan'], 
                       default='ai4i2020', help='Dataset to use for training')
    parser.add_argument('--subset', default='FD001', help='Subset for NASA CMAPSS')
    
    args = parser.parse_args()
    
    pipeline = RealDatasetPipeline()
    
    # Dataset-specific parameters
    dataset_params = {}
    if args.dataset == 'nasa_cmapss':
        dataset_params['subset'] = args.subset
    
    pipeline.run_pipeline(args.dataset, **dataset_params)

if __name__ == "__main__":
    main()
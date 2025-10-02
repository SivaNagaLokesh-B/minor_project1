# fixed_real_data_pipeline.py
import logging
import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add current directory and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class FixedRealDataPipeline:
    def __init__(self):
        self.config = self._load_config()
        self.expected_ranges = None
    
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
        logger.info("Starting Predictive Maintenance Pipeline with %s", dataset_name)
        
        try:
            # Phase 1: Load Real Dataset
            logger.info("Phase 1: Loading Real Dataset")
            
            # Import dataset loader
            from data_processing.real_dataset_loader import RealDatasetLoader
            loader = RealDatasetLoader()
            dataset = loader.load_dataset(dataset_name, download=download)
            
            # Save dataset to standard paths
            self._save_dataset(dataset)
            
            # Phase 2: Data Processing
            logger.info("Phase 2: Data Processing")
            from data_processing.data_loader import DataManager
            data_manager = DataManager(self.config)
            processed_data = data_manager.clean_and_preprocess(dataset)
            
            # Phase 3: Feature Engineering
            logger.info("Phase 3: Feature Engineering")
            from feature_engineering.feature_pipeline import FeatureEngineer
            feature_engineer = FeatureEngineer({})
            features, labels = feature_engineer.create_features_and_labels(processed_data)
            
            # Log the actual columns in labels and features for debugging
            logger.info("Labels columns: %s", labels.columns.tolist() if hasattr(labels, 'columns') else "No columns found")
            logger.info("Features columns: %s", features.columns.tolist() if hasattr(features, 'columns') else "No columns found")
            logger.info("Labels type: %s", type(labels))
            
            # Calculate expected ranges for API validation from original dataset
            self._calculate_expected_ranges_from_original(dataset['sensor'])
            
            # Phase 4: Model Training with Proper Validation
            logger.info("Phase 4: Model Training")
            from training.trainer import ModelTrainer
            trainer = ModelTrainer({})
            
            # Check what type of labels we have and split accordingly
            if hasattr(labels, 'columns'):
                # Labels is a DataFrame - check for appropriate stratification column
                label_columns = labels.columns.tolist()
                logger.info("Available label columns: %s", label_columns)
                
                # Try to find an appropriate column for stratification
                stratify_col = None
                for col in ['failure', 'failure_imminent', 'target', 'label', 'failure_probability']:
                    if col in label_columns:
                        stratify_col = labels[col]
                        logger.info("Using '%s' for stratification", col)
                        break
                
                if stratify_col is not None:
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2, random_state=42, stratify=stratify_col
                    )
                else:
                    # No suitable column found, split without stratification
                    logger.warning("No suitable stratification column found. Splitting without stratification.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2, random_state=42
                    )
            else:
                # Labels is a Series or array
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42
                )
            
            logger.info("Training data shape: X_train=%s, X_test=%s", X_train.shape, X_test.shape)
            
            # Train on training set only
            ensemble_model = trainer.train_ensemble(X_train, y_train)
            
            # Phase 5: Validation on Test Set
            logger.info("Phase 5: Model Validation")
            validation_results = trainer.validate_model(ensemble_model, X_test, y_test)
            
            # Log proper metrics - handle both dict and float returns
            if isinstance(validation_results, dict):
                if 'mae' in validation_results:
                    logger.info("Test Set - MAE: %.2f", validation_results['mae'])
                if 'rmse' in validation_results:
                    logger.info("Test Set - RMSE: %.2f", validation_results['rmse'])
            else:
                # Assume it's a float value (MAE)
                logger.info("Test Set - MAE: %.2f", validation_results)
            
            # Phase 6: Deployment
            logger.info("Phase 6: Deployment Setup")
            from deployment.prediction_system import ProductionSystem
            production_system = ProductionSystem(ensemble_model, self.config, {})
            production_system.save_pipeline()
            
            # Save expected ranges for API validation
            self._save_expected_ranges()
            
            logger.info("Pipeline completed successfully with %s!", dataset_name)
            self._print_dataset_stats(dataset)
            
            return production_system
            
        except Exception as e:
            logger.error("Error in pipeline: %s", e)
            import traceback
            traceback.print_exc()
            raise
    
    def _calculate_expected_ranges_from_original(self, sensor_data):
        """Calculate expected input ranges for API validation from original sensor data"""
        # Map of expected API input names to possible dataset column names
        sensor_mappings = {
            'vibration': ['vibration', 'vibration_x', 'vibration_y', 'vibration_z', 'vib_x', 'vib_y', 'vib_z'],
            'temperature': ['temperature', 'temp', 'air_temperature', 'process_temperature'],
            'pressure': ['pressure', 'air_pressure', 'process_pressure'],
            'current': ['current', 'motor_current', 'torque'],
            'rpm': ['rpm', 'rotational_speed', 'rotation_speed'],
            'tool_wear': ['tool_wear', 'tool_wear_min', 'tool_wear_max']
        }
        
        self.expected_ranges = {}
        sensor_columns = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
        logger.info("Available sensor columns: %s", sensor_columns)
        
        for api_name, possible_names in sensor_mappings.items():
            found_column = None
            for possible_name in possible_names:
                if possible_name in sensor_columns:
                    found_column = possible_name
                    break
            
            if found_column:
                mean_val = sensor_data[found_column].mean()
                std_val = sensor_data[found_column].std()
                # Use ±3 standard deviations as expected range
                self.expected_ranges[api_name] = {
                    'min': max(0, mean_val - 3 * std_val),  # Ensure non-negative
                    'max': mean_val + 3 * std_val,
                    'mean': mean_val,
                    'std': std_val,
                    'original_column': found_column  # Store for reference
                }
                logger.info("   %s (from %s): [%.2f, %.2f]", api_name, found_column, 
                           self.expected_ranges[api_name]['min'], self.expected_ranges[api_name]['max'])
            else:
                logger.warning("No matching column found for '%s'. Tried: %s", api_name, possible_names)
    
    def _save_expected_ranges(self):
        """Save expected ranges to file for API use"""
        import json
        os.makedirs(self.config['model_paths']['saved_models'], exist_ok=True)
        ranges_file = os.path.join(self.config['model_paths']['saved_models'], 'expected_ranges.json')
        
        if not self.expected_ranges:
            logger.error("No expected ranges calculated to save!")
            return
        
        # Convert numpy types to Python types for JSON serialization
        serializable_ranges = {}
        for col, ranges in self.expected_ranges.items():
            serializable_ranges[col] = {
                'min': float(ranges['min']),
                'max': float(ranges['max']),
                'mean': float(ranges['mean']),
                'std': float(ranges['std'])
            }
        
        with open(ranges_file, 'w') as f:
            json.dump(serializable_ranges, f, indent=2)
        
        logger.info("✅ Expected ranges saved to: %s", ranges_file)
    
    def _save_dataset(self, dataset):
        """Save dataset to standard file paths"""
        # Ensure timestamps are properly formatted before saving
        for key in ['sensor', 'maintenance', 'operational']:
            if key in dataset and 'timestamp' in dataset[key].columns:
                dataset[key]['timestamp'] = pd.to_datetime(dataset[key]['timestamp'], errors='coerce')
        
        dataset['sensor'].to_csv(self.config['data_paths']['raw_data'], index=False)
        dataset['maintenance'].to_csv(self.config['data_paths']['maintenance_logs'], index=False)
        dataset['operational'].to_csv(self.config['data_paths']['operational_data'], index=False)
        
        logger.info("Dataset saved:")
        logger.info("   - Sensor data: %s records", len(dataset['sensor']))
        logger.info("   - Maintenance logs: %s records", len(dataset['maintenance']))
        logger.info("   - Operational data: %s records", len(dataset['operational']))
    
    def _print_dataset_stats(self, dataset):
        """Print dataset statistics"""
        sensor_data = dataset['sensor']
        
        logger.info("Dataset Statistics:")
        logger.info("   Equipment count: %s", sensor_data['equipment_id'].nunique())
        logger.info("   Total records: %s", len(sensor_data))
        
        # Check for various failure column names - handle both numeric and string columns
        failure_columns = [col for col in sensor_data.columns if 'failure' in col.lower() or 'fail' in col.lower()]
        if failure_columns:
            for col in failure_columns:
                try:
                    # Check if column is numeric
                    if sensor_data[col].dtype in [np.number, 'int64', 'float64']:
                        failure_rate = sensor_data[col].mean() * 100
                        logger.info("   %s rate: %.2f%%", col, failure_rate)
                    else:
                        # For string columns, count occurrences of failure vs no failure
                        unique_counts = sensor_data[col].value_counts()
                        total_records = len(sensor_data)
                        
                        # Calculate failure rate based on string values
                        failure_keywords = ['failure', 'fail', 'broken', 'malfunction']
                        failure_count = 0
                        
                        for key, count in unique_counts.items():
                            key_str = str(key).lower()
                            if any(fail_word in key_str for fail_word in failure_keywords) and 'no failure' not in key_str:
                                failure_count += count
                        
                        if total_records > 0:
                            failure_rate = (failure_count / total_records) * 100
                            logger.info("   %s rate: %.2f%% (based on string analysis)", col, failure_rate)
                            logger.info("   %s distribution: %s", col, dict(unique_counts.head(10)))  # Show top 10 values
                        else:
                            logger.info("   %s: No records to analyze", col)
                            
                except Exception as e:
                    logger.warning("   Could not calculate failure rate for %s: %s", col, e)
        
        if 'timestamp' in sensor_data.columns:
            logger.info("   Date range: %s to %s", sensor_data['timestamp'].min(), sensor_data['timestamp'].max())
        
        # Sensor statistics - only for numeric columns
        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not any(fail_term in col.lower() for fail_term in ['failure', 'fail', 'degradation']):
                try:
                    logger.info("   %s: mean=%.2f, std=%.2f", col, sensor_data[col].mean(), sensor_data[col].std())
                except Exception as e:
                    logger.warning("   Could not calculate stats for %s: %s", col, e)

def main():
    parser = argparse.ArgumentParser(description='Train with Real Predictive Maintenance Datasets')
    parser.add_argument('--dataset', 
                       choices=['ai4i2020', 'nasa_turbofan', 'bearing_vibration'],
                       default='ai4i2020',
                       help='Dataset to use for training')
    parser.add_argument('--no-download', action='store_true',
                       help='Use local data instead of downloading')
    
    args = parser.parse_args()
    
    pipeline = FixedRealDataPipeline()
    pipeline.run_pipeline(args.dataset, download=not args.no_download)

if __name__ == "__main__":
    main()
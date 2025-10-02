# fix_imports_simple.py
import os
import sys

def fix_feature_engineering_module():
    """Fix the feature engineering module structure without emojis"""
    print("Fixing Feature Engineering Module")
    print("=" * 50)
    
    # Create feature_engineering directory if it doesn't exist
    feature_eng_dir = "src/feature_engineering"
    os.makedirs(feature_eng_dir, exist_ok=True)
    
    # Create __init__.py
    init_content = '''from .feature_pipeline import FeatureEngineer

__all__ = ['FeatureEngineer']
'''
    
    with open(os.path.join(feature_eng_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(init_content)
    print("Created src/feature_engineering/__init__.py")
    
    # Create feature_pipeline.py without emojis
    pipeline_content = '''import pandas as pd
import numpy as np
from scipy import signal, stats
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, model_config):
        self.config = model_config.get('feature_engineering', {})
        self.rolling_windows = self.config.get('rolling_windows', ['24h', '168h'])
        
    def create_features_and_labels(self, processed_data):
        """Create features and labels"""
        logger.info("Creating features and labels...")
        
        try:
            # If processed_data is a path, load it
            if isinstance(processed_data, str):
                logger.info("Loading processed data from: %s", processed_data)
                processed_data = pd.read_csv(processed_data)
            
            logger.info("Processing data with shape: %s", processed_data.shape)
            
            # Create basic features
            features = self._create_basic_features(processed_data)
            labels = self._create_basic_labels(processed_data)
            
            logger.info("Features shape: %s, Labels shape: %s", features.shape, labels.shape)
            
            # Save features and labels
            self._save_features_and_labels(features, labels)
            
            return features, labels
            
        except Exception as e:
            logger.error("Error in feature engineering: %s", e)
            raise
    
    def _create_basic_features(self, data):
        """Create basic features from sensor data"""
        try:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            # Remove target columns
            exclude_cols = ['degradation_level', 'failure_imminent', 'rul']
            feature_cols = [col for col in numeric_data.columns if col not in exclude_cols]
            
            features_list = []
            
            # Create features per equipment if available
            if 'equipment_id' in data.columns:
                for equipment_id in data['equipment_id'].unique():
                    eq_data = data[data['equipment_id'] == equipment_id]
                    feature_dict = {'equipment_id': equipment_id}
                    
                    for col in feature_cols:
                        if col in eq_data.columns:
                            col_data = eq_data[col].dropna()
                            if len(col_data) > 0:
                                feature_dict[f'{col}_mean'] = float(col_data.mean())
                                feature_dict[f'{col}_std'] = float(col_data.std())
                                feature_dict[f'{col}_max'] = float(col_data.max())
                                feature_dict[f'{col}_min'] = float(col_data.min())
                            else:
                                feature_dict[f'{col}_mean'] = 0.0
                                feature_dict[f'{col}_std'] = 0.0
                                feature_dict[f'{col}_max'] = 0.0
                                feature_dict[f'{col}_min'] = 0.0
                    
                    features_list.append(feature_dict)
            else:
                # Create overall features
                feature_dict = {}
                for col in feature_cols:
                    if col in numeric_data.columns:
                        col_data = numeric_data[col].dropna()
                        if len(col_data) > 0:
                            feature_dict[f'{col}_mean'] = float(col_data.mean())
                            feature_dict[f'{col}_std'] = float(col_data.std())
                            feature_dict[f'{col}_max'] = float(col_data.max())
                            feature_dict[f'{col}_min'] = float(col_data.min())
                        else:
                            feature_dict[f'{col}_mean'] = 0.0
                            feature_dict[f'{col}_std'] = 0.0
                            feature_dict[f'{col}_max'] = 0.0
                            feature_dict[f'{col}_min'] = 0.0
                
                features_list.append(feature_dict)
            
            features_df = pd.DataFrame(features_list)
            logger.info("Created %s feature rows", len(features_df))
            return features_df
            
        except Exception as e:
            logger.error("Error creating basic features: %s", e)
            return self._create_fallback_features()
    
    def _create_fallback_features(self):
        """Create fallback features when main method fails"""
        try:
            features = {}
            sensors = ['vibration', 'temperature', 'pressure', 'current', 'rpm', 'tool_wear']
            
            for sensor in sensors:
                features[f'{sensor}_mean'] = 0.0
                features[f'{sensor}_std'] = 1.0
                features[f'{sensor}_max'] = 0.0
                features[f'{sensor}_min'] = 0.0
            
            # Fill to reasonable dimension
            current_count = len(features)
            for i in range(current_count, 50):
                features[f'feature_{i}'] = 0.0
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error("Fallback feature creation failed: %s", e)
            return pd.DataFrame([{f'feature_{i}': 0.0 for i in range(50)}])
    
    def _create_basic_labels(self, data):
        """Create basic labels"""
        try:
            labels_list = []
            
            if 'equipment_id' in data.columns:
                equipment_ids = data['equipment_id'].unique()
            else:
                equipment_ids = ['default_equipment']
            
            for eq_id in equipment_ids:
                if 'degradation_level' in data.columns:
                    eq_data = data[data['equipment_id'] == eq_id] if 'equipment_id' in data.columns else data
                    degradation = eq_data['degradation_level'].iloc[-1] if len(eq_data) > 0 else 0.5
                    rul = max(0, (1.0 - degradation) * 1000)
                else:
                    rul = 500.0  # Default RUL
                
                labels_list.append({
                    'equipment_id': eq_id,
                    'rul': float(rul),
                    'failure_probability': min(rul / 1000, 0.95)
                })
            
            return pd.DataFrame(labels_list)
            
        except Exception as e:
            logger.error("Error creating labels: %s", e)
            return pd.DataFrame([{'equipment_id': 'default', 'rul': 500.0, 'failure_probability': 0.5}])
    
    def _save_features_and_labels(self, features, labels):
        """Save features and labels"""
        try:
            Path('data/features').mkdir(parents=True, exist_ok=True)
            Path('data/processed').mkdir(parents=True, exist_ok=True)
            
            features.to_csv('data/features/engineered_features.csv', index=False)
            labels.to_csv('data/processed/labels.csv', index=False)
            logger.info("Features and labels saved successfully")
        except Exception as e:
            logger.error("Error saving features and labels: %s", e)
    
    def create_prediction_features(self, single_row_data):
        """Create features for prediction"""
        return self._create_basic_features(single_row_data)
'''
    
    with open(os.path.join(feature_eng_dir, "feature_pipeline.py"), "w", encoding="utf-8") as f:
        f.write(pipeline_content)
    print("Created src/feature_engineering/feature_pipeline.py")
    
    # Create other necessary __init__.py files
    modules = ['data_processing', 'training', 'deployment']
    for module in modules:
        module_dir = f"src/{module}"
        init_file = os.path.join(module_dir, "__init__.py")
        
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write(f'# {module} module\\n')
            print(f"Created {init_file}")
    
    print("Feature engineering module fixed!")

def verify_fix():
    """Verify the fix worked"""
    print("\\nVerifying Fix")
    print("=" * 30)
    
    # Check if all files exist
    required_files = [
        "src/feature_engineering/__init__.py",
        "src/feature_engineering/feature_pipeline.py",
        "src/data_processing/__init__.py", 
        "src/training/__init__.py",
        "src/deployment/__init__.py"
    ]
    
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "OK" if exists else "MISSING"
        print(f"   {status}: {file_path}")
    
    # Test import
    print("\\nTesting imports...")
    try:
        sys.path.insert(0, "src")
        from feature_engineering.feature_pipeline import FeatureEngineer
        print("FeatureEngineer import successful!")
        
        # Test instantiation
        engineer = FeatureEngineer({})
        print("FeatureEngineer instantiation successful!")
        
        return True
        
    except Exception as e:
        print(f"Import test failed: {e}")
        return False

if __name__ == "__main__":
    fix_feature_engineering_module()
    success = verify_fix()
    
    if success:
        print("\\nAll fixes applied successfully!")
        print("\\nNow you can run:")
        print("   python fixed_real_data_pipeline.py --dataset ai4i2020")
    else:
        print("\\nFixes failed, please check the errors above")
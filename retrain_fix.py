# retrain_fix.py
import logging
import sys
import os
import pandas as pd
import joblib
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_retrain():
    """Quick retrain to fix all issues"""
    
    try:
        logger.info("🔄 Comprehensive retraining to fix all issues...")
        
        # Import required modules
        from data_processing.data_loader import DataManager
        from feature_engineering.feature_pipeline import FeatureEngineer
        from training.trainer import ModelTrainer
        from deployment.prediction_system import ProductionSystem
        
        # Load existing features and labels
        features_path = 'data/features/engineered_features.csv'
        labels_path = 'data/processed/labels.csv'
        
        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            logger.error("❌ Features or labels not found. Please run the full pipeline first.")
            return False
        
        features = pd.read_csv(features_path)
        labels = pd.read_csv(labels_path)
        
        logger.info(f"📊 Loaded features: {features.shape}, labels: {labels.shape}")
        
        # Clean features - remove any remaining NaN values
        features_clean = features.select_dtypes(include=[np.number]).fillna(0)
        logger.info(f"🧹 Cleaned features: {features_clean.shape}")
        
        # Retrain model with updated trainer
        trainer = ModelTrainer({})
        model = trainer.train_ensemble(features_clean, labels)
        
        if model is None:
            logger.error("❌ Model training failed")
            return False
        
        # Create and save production system
        config = {
            'data_paths': {
                'raw_data': 'data/raw/sensor_data.csv',
                'maintenance_logs': 'data/raw/maintenance_logs.csv',
                'operational_data': 'data/raw/operational_data.csv'
            }
        }
        
        production_system = ProductionSystem(model, config, {})
        production_system.save_pipeline()
        
        logger.info("✅ Comprehensive retraining completed successfully!")
        logger.info("✅ All training artifacts should now be available")
        
        # Verify everything was saved
        verify_training_artifacts()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick retraining failed: {e}")
        return False

def verify_training_artifacts():
    """Verify all training artifacts were created"""
    artifacts = {
        'Production Pipeline': 'models/production_pipeline.pkl',
        'Feature Scaler': 'models/scalers/feature_scaler.pkl',
        'Feature Info': 'models/training_info/feature_info.pkl',
        'Prediction Template': 'models/training_info/prediction_template.csv'
    }
    
    logger.info("🔍 Verifying training artifacts...")
    
    for name, path in artifacts.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        logger.info(f"   {status} {name}: {path}")

if __name__ == "__main__":
    success = quick_retrain()
    if success:
        print("\n🎯 Now test the production system:")
        print("   python test_production_fixed.py")
    else:
        print("\n💥 Quick retraining failed. Please check the errors above.")
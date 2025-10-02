# simple_retrain.py
import logging
import sys
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_retrain():
    """Simple retraining that doesn't depend on complex imports"""
    
    try:
        logger.info("Starting simple retraining...")
        
        # Check if we have the necessary data
        features_path = 'data/features/engineered_features.csv'
        labels_path = 'data/processed/labels.csv'
        
        if not os.path.exists(features_path):
            logger.error("Features not found. Need to run feature engineering first.")
            return False
        
        if not os.path.exists(labels_path):
            logger.error("Labels not found. Need to run feature engineering first.")
            return False
        
        # Load features and labels
        features = pd.read_csv(features_path)
        labels = pd.read_csv(labels_path)
        
        logger.info("Loaded features: %s, labels: %s", features.shape, labels.shape)
        
        # Clean data - remove non-numeric and handle NaN
        X = features.select_dtypes(include=[np.number]).fillna(0)
        
        if 'rul' not in labels.columns:
            logger.error("RUL column not found in labels")
            return False
        
        y = labels['rul'].values
        
        logger.info("Cleaned data: X=%s, y=%s", X.shape, y.shape)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        logger.info("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Validate
        from sklearn.metrics import mean_absolute_error
        predictions = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)
        
        logger.info("Model trained. Validation MAE: %.2f", mae)
        
        # Save model and artifacts
        os.makedirs('models/scalers', exist_ok=True)
        os.makedirs('models/training_info', exist_ok=True)
        
        # Save scaler
        joblib.dump(scaler, 'models/scalers/feature_scaler.pkl')
        logger.info("Scaler saved")
        
        # Save feature information
        feature_info = {
            'feature_columns': X.columns.tolist(),
            'feature_count': len(X.columns),
            'training_samples': len(X)
        }
        joblib.dump(feature_info, 'models/training_info/feature_info.pkl')
        logger.info("Feature info saved: %s features", len(X.columns))
        
        # Save prediction template
        template = {col: 0.0 for col in X.columns}
        template_df = pd.DataFrame([template])
        template_df.to_csv('models/training_info/prediction_template.csv', index=False)
        logger.info("Prediction template saved")
        
        # Create and save production system
        from deployment.prediction_system import ProductionSystem
        
        config = {
            'data_paths': {
                'raw_data': 'data/raw/sensor_data.csv',
                'maintenance_logs': 'data/raw/maintenance_logs.csv',
                'operational_data': 'data/raw/operational_data.csv'
            }
        }
        
        production_system = ProductionSystem(model, config, {})
        production_system.save_pipeline()
        
        logger.info("Simple retraining completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error("Simple retraining failed: %s", e)
        return False

if __name__ == "__main__":
    success = simple_retrain()
    if success:
        print("Now test the production system:")
        print("   python test_production_fixed.py")
    else:
        print("Simple retraining failed")
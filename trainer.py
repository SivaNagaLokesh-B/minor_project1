# src/training/trainer.py
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import joblib
import os
import pandas as pd

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_config):
        self.config = model_config.get('training', {})
        self.scalers = {}
        
    def train_ensemble(self, features, labels):
        """Train the ensemble model"""
        logger.info("Training ensemble model...")
        
        try:
            # Prepare data
            if features is None or labels is None:
                logger.error("No features or labels provided for training")
                return None
                
            X, y = self._prepare_training_data(features, labels)
            
            if X is None or y is None:
                logger.error("Could not prepare training data")
                return None
            
            # Simple train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
            
            # Train model
            model = self._train_simple_model(X_train_scaled, y_train)
            
            # Validate model
            validation_score = self._validate_model(model, X_test_scaled, y_test)
            
            # Save scalers and feature information
            self._save_training_artifacts(features, model)
            
            logger.info(f"Model training completed. Validation MAE: {validation_score:.2f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return None
    
    def _prepare_training_data(self, features, labels):
        """Prepare features and labels for training"""
        try:
            # Remove non-numeric columns and handle missing values
            X = features.select_dtypes(include=[np.number]).fillna(0)
            
            # Ensure we have the RUL column in labels
            if 'rul' not in labels.columns:
                logger.error("RUL column not found in labels")
                return None, None
                
            y = labels['rul'].values
            
            logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
            return X.values, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def _train_simple_model(self, X_train, y_train):
        """Train a simple model"""
        from sklearn.ensemble import RandomForestRegressor
        
        logger.info("Training Random Forest model...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        return model
    
    def _validate_model(self, model, X_test, y_test):
        """Validate model performance"""
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        logger.info(f"Validation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return mae
    
    def _save_training_artifacts(self, features, model):
        """Save all training artifacts including feature information"""
        try:
            # Create directories if they don't exist
            os.makedirs('models/scalers', exist_ok=True)
            os.makedirs('models/training_info', exist_ok=True)
            
            # Save the feature scaler
            if 'features' in self.scalers:
                joblib.dump(self.scalers['features'], 'models/scalers/feature_scaler.pkl')
                logger.info("✅ Feature scaler saved successfully")
            
            # Save feature information for prediction
            feature_info = {
                'feature_columns': features.select_dtypes(include=[np.number]).columns.tolist(),
                'feature_count': len(features.select_dtypes(include=[np.number]).columns),
                'training_samples': len(features)
            }
            
            joblib.dump(feature_info, 'models/training_info/feature_info.pkl')
            logger.info(f"✅ Feature info saved: {feature_info['feature_count']} features")
            
            # Save feature template for prediction
            self._create_prediction_template(features)
            
        except Exception as e:
            logger.error(f"Error saving training artifacts: {e}")
    
    def _create_prediction_template(self, features):
        """Create a prediction feature template"""
        try:
            # Get numeric columns only
            numeric_features = features.select_dtypes(include=[np.number])
            
            # Create template with feature names and default values
            template = {col: 0.0 for col in numeric_features.columns}
            
            # Save template
            template_df = pd.DataFrame([template])
            template_path = 'models/training_info/prediction_template.csv'
            template_df.to_csv(template_path, index=False)
            
            logger.info(f"✅ Prediction template created: {template_path}")
            
        except Exception as e:
            logger.error(f"Error creating prediction template: {e}")
    
    def validate_model(self, model, features, labels):
        """Public method for model validation"""
        if model is None or features is None or labels is None:
            return {"error": "Invalid inputs for validation"}
        
        X, y = self._prepare_training_data(features, labels)
        if X is not None and y is not None:
            X_scaled = self.scalers['features'].transform(X)
            return self._validate_model(model, X_scaled, y)
        else:
            return {"error": "Could not prepare validation data"}
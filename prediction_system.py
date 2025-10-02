# src/deployment/prediction_system.py
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import logging
import os
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ProductionSystem:
    def __init__(self, ensemble_model, config, model_config):
        self.model = ensemble_model
        self.config = config
        self.model_config = model_config
        self.confidence_threshold = 0.9
        self.scaler = None
        self._load_scaler()
        
    def _load_scaler(self):
        """Load or create scaler"""
        try:
            scaler_path = "models/scalers/feature_scaler.pkl"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Scaler loaded successfully")
            else:
                logger.warning("❌ Scaler not found, creating new scaler")
                self.scaler = StandardScaler()
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            self.scaler = StandardScaler()
        
    def predict(self, new_sensor_data):
        """Make predictions on new sensor data"""
        logger.info("Making prediction on new data...")
        
        try:
            # Preprocess new data
            processed_data = self._preprocess_new_data(new_sensor_data)
            
            # Create features
            features = self._create_simple_features(processed_data)
            
            if features is None or features.empty:
                return {"error": "Could not create features from input data"}
            
            # Scale features
            if self.scaler is not None:
                try:
                    features_scaled = self.scaler.transform(features)
                except Exception as e:
                    logger.warning(f"Scaling failed, using raw features: {e}")
                    features_scaled = features.values
            else:
                features_scaled = features.values
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            
            # Add confidence and recommendations
            result = self._format_prediction_result(prediction, processed_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {"error": str(e)}
    
    def _preprocess_new_data(self, data):
        """Preprocess new incoming data"""
        processed = data.copy()
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        processed[numeric_cols] = processed[numeric_cols].fillna(0)
        return processed
    
    def _create_simple_features(self, data):
        """Create simple features from new data"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data) == 0:
                return None
                
            # Basic statistical features
            features = {}
            sensor_columns = ['vibration', 'temperature', 'pressure', 'current', 'rpm', 'tool_wear']
            
            for col in sensor_columns:
                if col in numeric_data.columns:
                    col_data = numeric_data[col]
                    features[f'{col}_mean'] = float(col_data.mean())
                    features[f'{col}_std'] = float(col_data.std())
                    features[f'{col}_max'] = float(col_data.max())
                    features[f'{col}_min'] = float(col_data.min())
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return None
    
    def _format_prediction_result(self, prediction, data):
        """Format prediction with business context"""
        try:
            # Handle different prediction formats
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                rul = float(prediction[0])
            else:
                rul = float(prediction)
                
            failure_prob = min(rul / 1000, 0.95) if rul > 0 else 0.95
            
            # Calculate confidence
            confidence = self._calculate_confidence(rul)
            
            # Get equipment ID
            equipment_id = 'unknown'
            if 'equipment_id' in data.columns and len(data) > 0:
                equipment_id = data['equipment_id'].iloc[0]
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'equipment_id': equipment_id,
                'predicted_rul_hours': round(rul, 2),
                'failure_probability': round(failure_prob, 4),
                'confidence': round(confidence, 4),
                'maintenance_recommendation': self._generate_recommendation(rul, failure_prob),
                'predicted_failure_date': (datetime.now() + timedelta(hours=rul)).isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error formatting prediction: {e}")
            return {"error": f"Could not format prediction result: {str(e)}"}
    
    def _calculate_confidence(self, rul):
        """Calculate prediction confidence"""
        if rul > 500:
            return 0.85
        elif rul > 100:
            return 0.75
        else:
            return 0.65
    
    def _generate_recommendation(self, rul, failure_prob):
        """Generate maintenance recommendation"""
        if failure_prob > 0.8 or rul < 24:
            return "CRITICAL: Schedule immediate maintenance"
        elif failure_prob > 0.6 or rul < 168:
            return "HIGH: Schedule maintenance within 1 week"
        elif failure_prob > 0.3 or rul < 720:
            return "MEDIUM: Plan maintenance within 1 month"
        else:
            return "LOW: Continue monitoring"
    
    def save_pipeline(self):
        """Save the complete pipeline"""
        try:
            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)
            
            # Save the pipeline
            joblib.dump(self, "models/production_pipeline.pkl")
            logger.info("✅ Pipeline saved successfully")
        except Exception as e:
            logger.error(f"Error saving pipeline: {e}")
    
    @staticmethod
    def load_pipeline():
        """Load saved pipeline"""
        try:
            return joblib.load("models/production_pipeline.pkl")
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            return None
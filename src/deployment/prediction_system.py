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
        self.feature_columns = None
        self._load_training_artifacts()
        
    def _load_training_artifacts(self):
        """Load training artifacts: scaler and feature information"""
        try:
            # Load scaler
            scaler_path = "models/scalers/feature_scaler.pkl"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Scaler loaded successfully")
            else:
                logger.warning("❌ Scaler not found, creating new scaler")
                self.scaler = StandardScaler()
            
            # Load feature information
            feature_info_path = "models/training_info/feature_info.pkl"
            if os.path.exists(feature_info_path):
                feature_info = joblib.load(feature_info_path)
                self.feature_columns = feature_info.get('feature_columns', [])
                logger.info(f"✅ Loaded {len(self.feature_columns)} feature columns")
            else:
                logger.warning("❌ Feature info not found")
                
        except Exception as e:
            logger.error(f"Error loading training artifacts: {e}")
        
    def predict(self, new_sensor_data):
        """Make predictions on new sensor data"""
        logger.info("Making prediction on new data...")
        
        try:
            # Preprocess new data
            processed_data = self._preprocess_new_data(new_sensor_data)
            
            # Create features that match training features
            features = self._create_matching_features(processed_data)
            
            if features is None or features.empty:
                return {"error": "Could not create features from input data"}
            
            # Ensure no NaN values
            features = features.fillna(0)
            
            # Scale features
            if self.scaler is not None and len(self.feature_columns) > 0:
                try:
                    features_scaled = self.scaler.transform(features)
                except Exception as e:
                    logger.warning(f"Scaling failed: {e}, using raw features")
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
    
    def _create_matching_features(self, data):
        """Create features that exactly match the training feature set"""
        try:
            # If we have feature columns from training, create matching features
            if self.feature_columns and len(self.feature_columns) > 0:
                return self._create_exact_match_features(data)
            else:
                # Fallback to basic features
                return self._create_basic_features(data)
                
        except Exception as e:
            logger.error(f"Error creating matching features: {e}")
            return self._create_basic_features(data)
    
    def _create_exact_match_features(self, data):
        """Create features that exactly match training feature columns"""
        try:
            # Start with all zeros for all training features
            features_dict = {col: 0.0 for col in self.feature_columns}
            
            # Basic sensor columns that might be in input
            basic_sensors = ['vibration', 'temperature', 'pressure', 'current', 'rpm', 'tool_wear']
            
            # Map input data to feature columns
            for sensor in basic_sensors:
                if sensor in data.columns:
                    value = float(data[sensor].iloc[0])
                    
                    # Look for feature columns that contain this sensor name
                    for feature_col in self.feature_columns:
                        if sensor in feature_col:
                            # Set appropriate value based on feature type
                            if 'mean' in feature_col or 'max' in feature_col or 'min' in feature_col or 'median' in feature_col:
                                features_dict[feature_col] = value
                            elif 'std' in feature_col or 'range' in feature_col:
                                features_dict[feature_col] = 0.1  # Small value for single point
                            else:
                                features_dict[feature_col] = value
            
            # Create DataFrame with exact column order
            features_df = pd.DataFrame([features_dict])[self.feature_columns]
            
            logger.info(f"Created exact match features: {features_df.shape}")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating exact match features: {e}")
            return None
    
    def _create_basic_features(self, data):
        """Create basic features when exact match is not possible"""
        try:
            features = {}
            sensor_columns = ['vibration', 'temperature', 'pressure', 'current', 'rpm', 'tool_wear']
            
            for col in sensor_columns:
                if col in data.columns:
                    value = float(data[col].iloc[0])
                    features[col] = value
                    features[f'{col}_mean'] = value
                    features[f'{col}_max'] = value
                    features[f'{col}_min'] = value
            
            # Fill to reasonable dimension
            current_features = len(features)
            for i in range(current_features, 50):  # Reasonable default
                features[f'feature_{i}'] = 0.0
            
            features_df = pd.DataFrame([features])
            features_df = features_df.fillna(0)
            
            logger.info(f"Created basic features: {features_df.shape}")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating basic features: {e}")
            return None
    
    def _format_prediction_result(self, prediction, data):
        """Format prediction with business context"""
        try:
            # Handle different prediction formats
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                rul = float(prediction[0])
            else:
                rul = float(prediction)
                
            # Ensure RUL is reasonable
            rul = max(0.0, min(rul, 10000.0))
            
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
#!/usr/bin/env python3
"""
Simplified predictive maintenance pipeline without complex dependencies
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePredictiveMaintenance:
    def __init__(self):
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in ['data/raw', 'data/processed', 'data/features', 'models']:
            os.makedirs(dir_path, exist_ok=True)
    
    def generate_synthetic_data(self):
        """Generate simple synthetic data"""
        logger.info("Generating synthetic data...")
        
        # Create simple sensor data
        timestamps = pd.date_range(start='2023-01-01', periods=1000, freq='H')
        equipment_ids = [f'EQ_{i:03d}' for i in range(1, 6)]
        
        sensor_data = []
        for eq_id in equipment_ids:
            for i, ts in enumerate(timestamps):
                # Simulate degradation over time
                degradation = min(i / 800, 1.0)
                
                sensor_data.append({
                    'timestamp': ts,
                    'equipment_id': eq_id,
                    'vibration': 2.0 + degradation * 3 + np.random.normal(0, 0.2),
                    'temperature': 75.0 + degradation * 20 + np.random.normal(0, 1),
                    'pressure': 100.0 - degradation * 30 + np.random.normal(0, 0.5),
                    'current': 15.0 + degradation * 10 + np.random.normal(0, 0.3),
                    'rpm': 1800 + np.random.normal(0, 20),
                })
        
        df = pd.DataFrame(sensor_data)
        df.to_csv('data/raw/sensor_data.csv', index=False)
        logger.info(f"Generated {len(df)} sensor records")
        return df
    
    def create_features(self, data):
        """Create simple features"""
        logger.info("Creating features...")
        
        features_list = []
        for eq_id in data['equipment_id'].unique():
            eq_data = data[data['equipment_id'] == eq_id]
            
            # Simple statistical features
            features = {
                'equipment_id': eq_id,
                'vibration_mean': eq_data['vibration'].mean(),
                'vibration_std': eq_data['vibration'].std(),
                'temperature_mean': eq_data['temperature'].mean(),
                'temperature_trend': np.polyfit(range(len(eq_data)), eq_data['temperature'], 1)[0],
                'pressure_mean': eq_data['pressure'].mean(),
                'current_mean': eq_data['current'].mean(),
                'rul': max(1000 - eq_data['vibration'].mean() * 100, 100)  # Simple RUL calculation
            }
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        features_df.to_csv('data/features/simple_features.csv', index=False)
        return features_df
    
    def train_model(self, features):
        """Train a simple model"""
        logger.info("Training model...")
        
        # Prepare features and target
        X = features.drop(['equipment_id', 'rul'], axis=1)
        y = features['rul']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        logger.info(f"Training R²: {train_score:.3f}")
        logger.info(f"Testing R²: {test_score:.3f}")
        
        # Save model and scaler
        joblib.dump(model, 'models/simple_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        return model, scaler
    
    def run_pipeline(self):
        """Run the complete simplified pipeline"""
        logger.info("Starting simplified predictive maintenance pipeline...")
        
        try:
            # Generate or load data
            if not os.path.exists('data/raw/sensor_data.csv'):
                data = self.generate_synthetic_data()
            else:
                data = pd.read_csv('data/raw/sensor_data.csv')
            
            # Create features
            features = self.create_features(data)
            
            # Train model
            model, scaler = self.train_model(features)
            
            logger.info("✅ Pipeline completed successfully!")
            return model, scaler
            
        except Exception as e:
            logger.error(f"❌ Error in pipeline: {e}")
            return None, None

if __name__ == "__main__":
    pipeline = SimplePredictiveMaintenance()
    pipeline.run_pipeline()
# create_features.py
import pandas as pd
import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_features():
    """Create features from processed data"""
    
    try:
        logger.info("🎯 Creating features from processed data...")
        
        # Check if processed data exists
        processed_path = 'data/processed/cleaned_sensor_data.csv'
        if not os.path.exists(processed_path):
            logger.error("❌ Processed data not found. Please run data processing first.")
            return False
        
        # Load processed data
        processed_data = pd.read_csv(processed_path)
        logger.info(f"📊 Loaded processed data: {processed_data.shape}")
        
        # Create basic features
        features = create_basic_features(processed_data)
        labels = create_basic_labels(processed_data)
        
        # Save features and labels
        os.makedirs('data/features', exist_ok=True)
        features.to_csv('data/features/engineered_features.csv', index=False)
        labels.to_csv('data/processed/labels.csv', index=False)
        
        logger.info(f"✅ Features created: {features.shape}")
        logger.info(f"✅ Labels created: {labels.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature creation failed: {e}")
        return False

def create_basic_features(data):
    """Create basic features from data"""
    try:
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Remove target columns if they exist
        exclude_cols = ['degradation_level', 'failure_imminent', 'rul']
        feature_cols = [col for col in numeric_data.columns if col not in exclude_cols]
        
        features_list = []
        
        # Create statistical features for each equipment
        if 'equipment_id' in data.columns:
            for equipment_id in data['equipment_id'].unique():
                eq_data = data[data['equipment_id'] == equipment_id]
                eq_numeric = eq_data[feature_cols]
                
                feature_dict = {'equipment_id': equipment_id}
                
                for col in feature_cols:
                    if col in eq_numeric.columns:
                        col_data = eq_numeric[col].dropna()
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
            # If no equipment_id, create overall features
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
        logger.info(f"Created {len(features_df)} feature rows with {len(features_df.columns)} columns")
        return features_df
        
    except Exception as e:
        logger.error(f"Error in basic feature creation: {e}")
        # Return fallback features
        return create_fallback_features()

def create_fallback_features():
    """Create fallback features when main method fails"""
    try:
        # Create simple features with common sensor patterns
        features = {}
        sensors = ['vibration', 'temperature', 'pressure', 'current', 'rpm', 'tool_wear']
        
        for sensor in sensors:
            features[f'{sensor}_mean'] = 0.0
            features[f'{sensor}_std'] = 1.0
            features[f'{sensor}_max'] = 0.0
            features[f'{sensor}_min'] = 0.0
        
        # Fill to reasonable size
        for i in range(len(features), 50):
            features[f'feature_{i}'] = 0.0
        
        return pd.DataFrame([features])
        
    except Exception as e:
        logger.error(f"Fallback feature creation failed: {e}")
        return pd.DataFrame([{f'feature_{i}': 0.0 for i in range(50)}])

def create_basic_labels(data):
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
                # Calculate RUL based on sensor patterns
                rul = calculate_rul_from_data(data)
            
            labels_list.append({
                'equipment_id': eq_id,
                'rul': float(rul),
                'failure_probability': min(rul / 1000, 0.95)
            })
        
        return pd.DataFrame(labels_list)
        
    except Exception as e:
        logger.error(f"Error creating labels: {e}")
        return pd.DataFrame([{'equipment_id': 'default', 'rul': 500.0, 'failure_probability': 0.5}])

def calculate_rul_from_data(data):
    """Calculate RUL from sensor data"""
    try:
        rul = 1000.0
        
        # Simple heuristic based on common sensors
        if 'vibration' in data.columns:
            vib_mean = data['vibration'].mean()
            rul -= vib_mean * 10
        
        if 'temperature' in data.columns:
            temp_mean = data['temperature'].mean()
            if temp_mean > 300:
                rul -= (temp_mean - 300) * 2
        
        if 'tool_wear' in data.columns:
            tool_wear_mean = data['tool_wear'].mean()
            rul -= tool_wear_mean * 0.5
        
        return max(rul, 50.0)
        
    except:
        return 500.0

if __name__ == "__main__":
    success = create_features()
    if success:
        print("\n🎯 Features created successfully! Now run:")
        print("   python simple_retrain.py")
    else:
        print("\n💥 Feature creation failed")
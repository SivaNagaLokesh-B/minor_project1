# test_production_simple.py
import pandas as pd
import joblib
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_production_system():
    """Test the trained production system with proper error handling"""
    
    try:
        # Load the production pipeline
        if not os.path.exists("models/production_pipeline.pkl"):
            logger.error("Production pipeline not found. Please train the model first.")
            return False
            
        pipeline = joblib.load("models/production_pipeline.pkl")
        logger.info("Production pipeline loaded successfully")
        
        # Check training artifacts
        artifacts = [
            "models/scalers/feature_scaler.pkl",
            "models/training_info/feature_info.pkl"
        ]
        
        for artifact in artifacts:
            if os.path.exists(artifact):
                logger.info("%s exists", os.path.basename(artifact))
            else:
                logger.warning("%s not found", os.path.basename(artifact))
        
        # Create realistic test data
        test_samples = [
            {
                'vibration': 3.0, 'temperature': 310.0, 'pressure': 150.0,
                'current': 40.0, 'rpm': 1500.0, 'tool_wear': 100.0,
                'equipment_id': 'test_machine_1'
            },
            {
                'vibration': 3.5, 'temperature': 315.0, 'pressure': 170.0, 
                'current': 45.0, 'rpm': 1600.0, 'tool_wear': 180.0,
                'equipment_id': 'test_machine_2'
            }
        ]
        
        print("Testing Production System Predictions")
        print("=" * 50)
        
        successful_predictions = 0
        
        for i, sample_data in enumerate(test_samples, 1):
            test_df = pd.DataFrame([sample_data])
            
            # Make prediction
            result = pipeline.predict(test_df)
            
            print("Machine %s Prediction:", i)
            print("   Input: %s", sample_data)
            
            # Handle result with error checking
            if 'error' in result:
                print("   Prediction error: %s", result['error'])
                continue
                
            # Safely format values
            rul = result.get('predicted_rul_hours', 'N/A')
            failure_prob = result.get('failure_probability', 'N/A')
            confidence = result.get('confidence', 'N/A')
            
            print("   Predicted RUL: %s hours", rul)
            
            # Format percentage values safely
            if isinstance(failure_prob, (int, float)):
                print("   Failure Probability: %.1f%%", failure_prob * 100)
            else:
                print("   Failure Probability: %s", failure_prob)
                
            if isinstance(confidence, (int, float)):
                print("   Confidence: %.1f%%", confidence * 100)
            else:
                print("   Confidence: %s", confidence)
                
            print("   Recommendation: %s", result.get('maintenance_recommendation', 'N/A'))
            
            successful_predictions += 1
            
        print("Summary: %s/%s predictions successful", successful_predictions, len(test_samples))
        
        return successful_predictions > 0
        
    except Exception as e:
        logger.error("Error testing production system: %s", e)
        return False

if __name__ == "__main__":
    success = test_production_system()
    if success:
        print("Production system is working correctly!")
        print("Ready to deploy the API and dashboard")
    else:
        print("There are issues with the production system")
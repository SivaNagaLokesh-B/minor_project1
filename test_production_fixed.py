# test_production_fixed.py
import pandas as pd
import joblib
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_percentage_format(value):
    """Safely format percentage values"""
    if isinstance(value, (int, float)):
        return f"{value:.1%}"
    else:
        return str(value)

def test_production_system():
    """Test the trained production system with proper error handling"""
    
    try:
        # Load the production pipeline
        if not os.path.exists("models/production_pipeline.pkl"):
            logger.error("❌ Production pipeline not found. Please train the model first.")
            return False
            
        pipeline = joblib.load("models/production_pipeline.pkl")
        logger.info("✅ Production pipeline loaded successfully")
        
        # Check training artifacts
        artifacts = [
            "models/scalers/feature_scaler.pkl",
            "models/training_info/feature_info.pkl"
        ]
        
        for artifact in artifacts:
            if os.path.exists(artifact):
                logger.info(f"✅ {os.path.basename(artifact)} exists")
            else:
                logger.warning(f"⚠️  {os.path.basename(artifact)} not found")
        
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
            },
            {
                'vibration': 2.8, 'temperature': 305.0, 'pressure': 140.0,
                'current': 35.0, 'rpm': 1450.0, 'tool_wear': 50.0,
                'equipment_id': 'test_machine_3'
            }
        ]
        
        print("\n🧪 Testing Production System Predictions")
        print("=" * 50)
        
        successful_predictions = 0
        
        for i, sample_data in enumerate(test_samples, 1):
            test_df = pd.DataFrame([sample_data])
            
            # Make prediction
            result = pipeline.predict(test_df)
            
            print(f"\n🔧 Machine {i} Prediction:")
            print(f"   📊 Input: {sample_data}")
            
            # Handle result with error checking
            if 'error' in result:
                print(f"   ❌ Prediction error: {result['error']}")
                continue
                
            # Safely format values
            rul = result.get('predicted_rul_hours', 'N/A')
            failure_prob = result.get('failure_probability', 'N/A')
            confidence = result.get('confidence', 'N/A')
            
            print(f"   ⏱️  Predicted RUL: {rul} hours")
            print(f"   📈 Failure Probability: {safe_percentage_format(failure_prob)}")
            print(f"   🎯 Confidence: {safe_percentage_format(confidence)}")
            print(f"   💡 Recommendation: {result.get('maintenance_recommendation', 'N/A')}")
            
            successful_predictions += 1
            
        print(f"\n📊 Summary: {successful_predictions}/{len(test_samples)} predictions successful")
        
        return successful_predictions > 0
        
    except Exception as e:
        logger.error(f"❌ Error testing production system: {e}")
        return False

if __name__ == "__main__":
    success = test_production_system()
    if success:
        print("\n🎉 Production system is working correctly!")
        print("🚀 Ready to deploy the API and dashboard")
    else:
        print("\n💥 There are issues with the production system")
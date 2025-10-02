# comprehensive_diagnostic.py
import os
import sys
import pandas as pd
import joblib

def comprehensive_diagnostic():
    print("🔍 Comprehensive Diagnostic")
    print("=" * 50)
    
    # 1. Check directories
    print("\n1. 📁 Directory Structure:")
    directories = [
        'models/training_info',
        'models/scalers', 
        'data/features',
        'data/processed',
        'data/raw'
    ]
    
    for directory in directories:
        exists = os.path.exists(directory)
        status = "✅" if exists else "❌"
        print(f"   {status} {directory}")
    
    # 2. Check training artifacts
    print("\n2. 📊 Training Artifacts:")
    artifacts = [
        'models/production_pipeline.pkl',
        'models/scalers/feature_scaler.pkl',
        'models/training_info/feature_info.pkl',
        'models/training_info/prediction_template.csv',
        'data/features/engineered_features.csv',
        'data/processed/labels.csv'
    ]
    
    for artifact in artifacts:
        exists = os.path.exists(artifact)
        status = "✅" if exists else "❌"
        print(f"   {status} {artifact}")
        
        # Additional info for key files
        if exists and artifact.endswith('.csv'):
            try:
                df = pd.read_csv(artifact)
                print(f"      📈 Shape: {df.shape}")
            except:
                print(f"      ❌ Could not read CSV")
        elif exists and artifact.endswith('.pkl'):
            try:
                obj = joblib.load(artifact)
                print(f"      🔧 Type: {type(obj).__name__}")
            except:
                print(f"      ❌ Could not load pickle")
    
    # 3. Check feature consistency
    print("\n3. 🎯 Feature Consistency:")
    try:
        if os.path.exists('models/training_info/feature_info.pkl'):
            feature_info = joblib.load('models/training_info/feature_info.pkl')
            feature_count = feature_info.get('feature_count', 'Unknown')
            print(f"   ✅ Training features: {feature_count}")
        else:
            print("   ❌ No feature info found")
            
        if os.path.exists('data/features/engineered_features.csv'):
            features_df = pd.read_csv('data/features/engineered_features.csv')
            print(f"   ✅ Actual features: {features_df.shape[1]}")
        else:
            print("   ❌ No features file found")
            
    except Exception as e:
        print(f"   ❌ Error checking features: {e}")
    
    # 4. Test model loading
    print("\n4. 🤖 Model Loading Test:")
    try:
        if os.path.exists('models/production_pipeline.pkl'):
            pipeline = joblib.load('models/production_pipeline.pkl')
            print(f"   ✅ Pipeline loaded: {type(pipeline).__name__}")
            
            # Test prediction with simple data
            test_data = pd.DataFrame([{
                'vibration': 3.0, 'temperature': 310.0, 'pressure': 150.0,
                'current': 40.0, 'rpm': 1500.0, 'tool_wear': 100.0
            }])
            
            try:
                result = pipeline.predict(test_data)
                if 'error' in result:
                    print(f"   ❌ Prediction error: {result['error']}")
                else:
                    print(f"   ✅ Prediction successful: RUL={result.get('predicted_rul_hours', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
        else:
            print("   ❌ No pipeline found")
            
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
    
    print("\n🎯 Diagnostic completed!")

if __name__ == "__main__":
    comprehensive_diagnostic()
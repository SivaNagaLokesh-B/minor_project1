# run_complete_pipeline.py
import os
import subprocess
import sys
import time

def run_command(command, description):
    """Run a shell command and print output"""
    print(f"\n🎯 {description}")
    print(f"   💻 Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ SUCCESS")
            if result.stdout:
                print(f"   📋 Output: {result.stdout[:200]}...")
            return True
        else:
            print(f"   ❌ FAILED")
            print(f"   💥 Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return False

def main():
    print("🚀 Complete Predictive Maintenance Pipeline")
    print("=" * 50)
    
    # Step 1: Create missing directory
    os.makedirs('models/training_info', exist_ok=True)
    print("✅ Created models/training_info directory")
    
    # Step 2: Run retraining
    commands = [
        ("python retrain_fix.py", "Running comprehensive retraining"),
        ("python test_production_fixed.py", "Testing production system"),
    ]
    
    success = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            success = False
            break
    
    if success:
        print("\n🎉 Pipeline execution completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Terminal 1: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload")
        print("   2. Terminal 2: python test_api.py")
        print("   3. Terminal 3: streamlit run dashboard.py")
    else:
        print("\n💥 Pipeline execution failed!")
        print("   🔧 Run: python comprehensive_diagnostic.py to identify issues")

if __name__ == "__main__":
    main()
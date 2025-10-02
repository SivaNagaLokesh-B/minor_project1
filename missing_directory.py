# create_missing_dir.py
import os

def create_missing_directory():
    missing_dir = 'models/training_info'
    try:
        os.makedirs(missing_dir, exist_ok=True)
        print(f"✅ Created missing directory: {missing_dir}")
        
        # Verify it was created
        if os.path.exists(missing_dir):
            print("✅ Directory verification: SUCCESS")
        else:
            print("❌ Directory verification: FAILED")
            
    except Exception as e:
        print(f"❌ Error creating directory: {e}")

if __name__ == "__main__":
    create_missing_directory()
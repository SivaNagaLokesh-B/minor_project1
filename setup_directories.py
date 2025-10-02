# setup_directories.py
import os

def setup_project_directories():
    """Create all necessary directories for the project"""
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/features',
        'models/scalers',
        'models/training_info',
        'logs',
        'src/data_processing',
        'src/feature_engineering',
        'src/training',
        'src/deployment'
    ]
    
    print("📁 Setting up project directories...")
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}")
        except Exception as e:
            print(f"❌ Failed to create {directory}: {e}")
    
    print("\n🎯 Directory setup completed!")

if __name__ == "__main__":
    setup_project_directories()
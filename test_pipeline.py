#!/usr/bin/env python3
"""
Test script to check if all components work
"""

import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment():
    """Test if the environment is set up correctly"""
    logger.info("Testing pipeline environment...")
    
    # Check if required directories exist
    required_dirs = ['data/raw', 'data/processed', 'data/features', 'models', 'src']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.warning(f"Directory missing: {dir_path}")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        else:
            logger.info(f"Directory exists: {dir_path}")
    
    # Check if config files exist
    config_files = ['config/paths.yaml', 'config/model_params.yaml']
    for config_file in config_files:
        if not os.path.exists(config_file):
            logger.error(f"Config file missing: {config_file}")
            return False
        else:
            logger.info(f"Config file exists: {config_file}")
    
    logger.info("✅ Environment test passed!")
    return True

if __name__ == "__main__":
    if test_environment():
        print("Environment is ready. Run: python run_pipeline.py")
    else:
        print("Environment setup failed. Please check the errors above.")
#!/usr/bin/env python3
"""
Quick start script for the predictive maintenance pipeline
"""

import sys
import logging
from main import PredictiveMaintenancePipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def quick_start():
    """Run the pipeline with default settings"""
    logger.info("Starting Predictive Maintenance Pipeline...")
    
    try:
        pipeline = PredictiveMaintenancePipeline()
        production_system = pipeline.run_full_pipeline()
        
        logger.info("✅ Pipeline completed successfully!")
        logger.info("📊 Models saved in: models/")
        logger.info("📈 Features saved in: data/features/")
        logger.info("🚀 Production system ready for predictions")
        
        return production_system
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    quick_start()
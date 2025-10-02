import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config):
        self.config = config
        self.data_paths = config['data_paths']
        
    def load_all_data(self):
        """Load all required data sources"""
        logger.info("Loading raw data...")
        
        data = {}
        try:
            # Check if files exist
            for data_type, path in self.data_paths.items():
                if not os.path.exists(path):
                    logger.error(f"Data file not found: {path}")
                    raise FileNotFoundError(f"Data file not found: {path}")
            
            data['sensor'] = pd.read_csv(self.data_paths['raw_data'])
            data['maintenance'] = pd.read_csv(self.data_paths['maintenance_logs'])
            data['operational'] = pd.read_csv(self.data_paths['operational_data'])
            
            # Convert timestamp columns
            for key in data:
                if 'timestamp' in data[key].columns:
                    data[key]['timestamp'] = pd.to_datetime(data[key]['timestamp'])
            
            logger.info(f"Loaded {len(data['sensor'])} sensor records")
            logger.info(f"Loaded {len(data['maintenance'])} maintenance records")
            logger.info(f"Loaded {len(data['operational'])} operational records")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
        return data
    
    def clean_and_preprocess(self, data):
        """Clean and preprocess the raw data"""
        logger.info("Cleaning and preprocessing data...")
        
        try:
            # Sensor data cleaning
            sensor_data = data['sensor'].copy()
            
            # Handle missing values
            sensor_data = self._handle_missing_values(sensor_data)
            
            # Remove outliers using IQR method
            sensor_data = self._remove_outliers(sensor_data)
            
            # Merge with operational data
            processed_data = self._merge_datasets(sensor_data, data['operational'])
            
            # Save processed data as CSV instead of Parquet
            processed_data_path = self.config['processed_paths']['cleaned_data'].replace('.parquet', '.csv')
            processed_data.to_csv(processed_data_path, index=False)
            
            logger.info(f"Processed data saved to: {processed_data_path}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def _handle_missing_values(self, df, method='interpolate'):
        """Handle missing values in sensor data"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        missing_before = df[numeric_columns].isnull().sum().sum()
        
        if missing_before > 0:
            if method == 'interpolate':
                df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
                df[numeric_columns] = df[numeric_columns].fillna(method='bfill')
            else:
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        missing_after = df[numeric_columns].isnull().sum().sum()
        logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
            
        return df
    
    def _remove_outliers(self, df, threshold=3):
        """Remove outliers using Z-score method"""
        from scipy import stats
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        initial_count = len(df)
        
        for col in numeric_columns:
            # Only remove outliers if we have enough data
            if len(df[col].dropna()) > 10:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                # Keep rows where z-score is below threshold or NaN (for missing values)
                valid_indices = (z_scores < threshold) | np.isnan(z_scores)
                df = df[df.index.isin(df[col].dropna().index[valid_indices])]
        
        final_count = len(df)
        removed_count = initial_count - final_count
        if removed_count > 0:
            logger.info(f"Outliers removed: {initial_count} -> {final_count} records ({removed_count} removed)")
        else:
            logger.info(f"No outliers removed. Record count: {initial_count}")
            
        return df
    
    def _merge_datasets(self, sensor_data, operational_data):
        """Merge sensor data with operational context"""
        try:
            # Sort by timestamp first
            sensor_sorted = sensor_data.sort_values('timestamp')
            operational_sorted = operational_data.sort_values('timestamp')
            
            # Ensure we have the required columns
            if 'equipment_id' not in sensor_sorted.columns or 'equipment_id' not in operational_sorted.columns:
                logger.warning("equipment_id column missing. Cannot merge datasets.")
                return sensor_sorted
            
            merged_data = pd.merge_asof(
                sensor_sorted,
                operational_sorted,
                on='timestamp',
                by='equipment_id'
            )
            
            logger.info(f"Merged data shape: {merged_data.shape}")
            return merged_data
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            # Return original sensor data if merge fails
            return sensor_data
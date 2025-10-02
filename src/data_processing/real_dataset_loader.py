import pandas as pd
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RealDatasetLoader:
    def __init__(self):
        self.dataset_info = {
            'nasa_turbofan': {
                'url': 'https://ti.arc.nasa.gov/c/6/',
                'description': 'NASA Turbofan Engine Degradation Simulation',
                'features': ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'],
                'target': 'RUL'
            },
            'ai4i2020': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv',
                'description': 'AI4I 2020 Predictive Maintenance Dataset',
                'features': ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'],
                'target': 'Machine failure'
            },
            'bearing_vibration': {
                'url': 'https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset',
                'description': 'Bearing Vibration Data for Fault Diagnosis',
                'features': ['horizontal_vibration', 'vertical_vibration', 'radial_vibration'],
                'target': 'failure_label'
            }
        }
    
    def load_dataset(self, dataset_name, download=True):
        """Load real predictive maintenance dataset"""
        logger.info(f"Loading {dataset_name} dataset...")
        
        if dataset_name not in self.dataset_info:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(self.dataset_info.keys())}")
        
        if dataset_name == 'ai4i2020':
            return self._load_ai4i2020(download)
        elif dataset_name == 'nasa_turbofan':
            return self._load_nasa_turbofan(download)
        elif dataset_name == 'bearing_vibration':
            return self._load_bearing_data(download)
        else:
            raise ValueError(f"Dataset loader for {dataset_name} not implemented")
    
    def _load_ai4i2020(self, download=True):
        """Load AI4I 2020 Predictive Maintenance Dataset"""
        try:
            if download:
                # Download from UCI repository
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
                df = pd.read_csv(url)
                logger.info(f"Downloaded AI4I 2020 dataset with {len(df)} records")
            else:
                # Load from local file
                df = pd.read_csv('data/raw/ai4i2020.csv')
            
            # Convert to our pipeline format
            return self._convert_ai4i_to_our_format(df)
            
        except Exception as e:
            logger.error(f"Error loading AI4I dataset: {e}")
            # Fallback to synthetic data
            return self._create_ai4i_synthetic_data()
    
    def _convert_ai4i_to_our_format(self, df):
        """Convert AI4I dataset to our standard format with FIXED timestamp generation"""
        sensor_data = []
        maintenance_data = []
        
        # Use proper date generation without invalid dates
        start_date = pd.Timestamp('2020-01-01')
        
        for idx, row in df.iterrows():
            # Create valid timestamp from UDI (hours from start)
            hours = int(row['UDI'])
            timestamp = start_date + pd.Timedelta(hours=hours)
            
            sensor_data.append({
                'timestamp': timestamp,
                'equipment_id': f"Machine_{row['Product ID'].replace('M', '')}",
                'vibration': row['Air temperature [K]'] / 100,
                'temperature': row['Process temperature [K]'],
                'pressure': row['Rotational speed [rpm]'] / 10,
                'current': row['Torque [Nm]'],
                'rpm': row['Rotational speed [rpm]'],
                'tool_wear': row['Tool wear [min]'],
                'degradation_level': min(row['Tool wear [min]'] / 200, 1.0),
                'failure_imminent': row['Machine failure'] == 1,
                'failure_type': self._get_failure_type(row)
            })
            
            # Create maintenance record for failures
            if row['Machine failure'] == 1:
                maintenance_data.append({
                    'timestamp': timestamp,
                    'equipment_id': f"Machine_{row['Product ID'].replace('M', '')}",
                    'maintenance_type': 'corrective',
                    'description': f"Failure: {self._get_failure_type(row)}",
                    'parts_replaced': 'bearings, tools',
                    'technician': 'Tech_01',
                    'downtime_hours': np.random.randint(4, 24)
                })
        
        # Create operational data
        operational_data = self._create_operational_data(sensor_data)
        
        return {
            'sensor': pd.DataFrame(sensor_data),
            'maintenance': pd.DataFrame(maintenance_data),
            'operational': operational_data
        }
    
    def _get_failure_type(self, row):
        """Extract failure type from AI4I dataset"""
        failure_types = {
            'TWF': 'Tool Wear Failure',
            'HDF': 'Heat Dissipation Failure', 
            'PWF': 'Power Failure',
            'OSF': 'Overstrain Failure',
            'RNF': 'Random Failure'
        }
        
        for ft, description in failure_types.items():
            if row[ft] == 1:
                return description
        return 'No Failure'
    
    def _create_operational_data(self, sensor_data):
        """Create operational context data"""
        equipment_ids = list(set([item['equipment_id'] for item in sensor_data]))
        
        operational_data = []
        for eq_id in equipment_ids:
            operational_data.append({
                'timestamp': pd.Timestamp('2020-01-01 00:00:00'),
                'equipment_id': eq_id,
                'production_rate': np.random.uniform(80, 100),
                'shift': np.random.choice(['A', 'B', 'C']),
                'operator_id': f"OP_{np.random.randint(1, 20):03d}",
                'ambient_temperature': np.random.uniform(15, 35),
                'ambient_humidity': np.random.uniform(30, 80)
            })
        
        return pd.DataFrame(operational_data)
    
    def _load_nasa_turbofan(self, download=True):
        """Load NASA Turbofan dataset (placeholder - would require actual download)"""
        logger.info("NASA Turbofan dataset requires manual download")
        logger.info("Please download from: https://ti.arc.nasa.gov/c/6/")
        logger.info("Using AI4I dataset as fallback...")
        return self._load_ai4i2020(download)
    
    def _load_bearing_data(self, download=True):
        """Load bearing vibration data (placeholder)"""
        logger.info("Bearing dataset requires manual download")
        logger.info("Using AI4I dataset as fallback...")
        return self._load_ai4i2020(download)
    
    def _create_ai4i_synthetic_data(self):
        """Create synthetic data in AI4I format as fallback"""
        logger.info("Creating synthetic AI4I-style data...")
        
        # Generate synthetic data that mimics AI4I structure
        num_machines = 100
        records_per_machine = 100
        start_date = pd.Timestamp('2020-01-01')
        
        sensor_data = []
        maintenance_data = []
        
        for machine_id in range(1, num_machines + 1):
            for hour in range(records_per_machine):
                timestamp = start_date + pd.Timedelta(hours=hour)
                tool_wear = min(hour * 2, 200)  # Progressive tool wear
                
                sensor_data.append({
                    'timestamp': timestamp,
                    'equipment_id': f"Machine_{machine_id}",
                    'vibration': 2.0 + (tool_wear / 200) * 3 + np.random.normal(0, 0.2),
                    'temperature': 300 + (tool_wear / 200) * 20 + np.random.normal(0, 2),
                    'pressure': 100.0 - (tool_wear / 200) * 30 + np.random.normal(0, 0.5),
                    'current': 40.0 + (tool_wear / 200) * 10 + np.random.normal(0, 0.3),
                    'rpm': 1500 + np.random.normal(0, 20),
                    'tool_wear': tool_wear,
                    'degradation_level': tool_wear / 200,
                    'failure_imminent': tool_wear > 180,
                    'failure_type': 'Tool Wear Failure' if tool_wear > 180 else 'No Failure'
                })
                
                if tool_wear > 180 and hour == records_per_machine - 1:
                    maintenance_data.append({
                        'timestamp': timestamp,
                        'equipment_id': f"Machine_{machine_id}",
                        'maintenance_type': 'corrective',
                        'description': 'Tool wear failure',
                        'parts_replaced': 'cutting tool',
                        'technician': 'Tech_01',
                        'downtime_hours': 8
                    })
        
        operational_data = self._create_operational_data(sensor_data)
        
        return {
            'sensor': pd.DataFrame(sensor_data),
            'maintenance': pd.DataFrame(maintenance_data),
            'operational': operational_data
        }
    
    def get_dataset_info(self, dataset_name):
        """Get information about available datasets"""
        if dataset_name in self.dataset_info:
            return self.dataset_info[dataset_name]
        else:
            return None
    
    def list_datasets(self):
        """List all available datasets"""
        return list(self.dataset_info.keys())
import pandas as pd
import numpy as np
import os
import requests
import zipfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self):
        self.datasets = {
            'nasa_cmapss': self.load_nasa_cmapss,
            'ai4i2020': self.load_ai4i2020,
            'bearing': self.load_bearing_data,
            'turbofan': self.load_turbofan_data
        }
    
    def load_dataset(self, dataset_name, **kwargs):
        """Main method to load any supported dataset"""
        if dataset_name in self.datasets:
            return self.datasets[dataset_name](**kwargs)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(self.datasets.keys())}")
    
    def load_nasa_cmapss(self, subset='FD001', download=True):
        """Load NASA CMAPSS Turbofan Engine Dataset"""
        logger.info(f"Loading NASA CMAPSS dataset - Subset {subset}")
        
        if download:
            self._download_nasa_data()
        
        # Load training data
        train_file = f"data/raw/CMAPSSData/train_{subset}.txt"
        test_file = f"data/raw/CMAPSSData/test_{subset}.txt"
        rul_file = f"data/raw/CMAPSSData/RUL_{subset}.txt"
        
        # Load and preprocess NASA data
        train_data = pd.read_csv(train_file, sep="\s+", header=None)
        test_data = pd.read_csv(test_file, sep="\s+", header=None)
        rul_data = pd.read_csv(rul_file, sep="\s+", header=None)
        
        # Add column names based on NASA documentation
        columns = ['unit', 'cycle'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
        train_data.columns = columns
        test_data.columns = columns
        
        return self._preprocess_nasa_data(train_data, test_data, rul_data)
    
    def load_ai4i2020(self):
        """Load AI4I 2020 Predictive Maintenance Dataset"""
        logger.info("Loading AI4I 2020 Predictive Maintenance Dataset")
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        data = pd.read_csv(url)
        
        # Preprocess for our pipeline
        processed_data = self._preprocess_ai4i_data(data)
        return processed_data
    
    def load_bearing_data(self):
        """Load Bearing Vibration Dataset"""
        logger.info("Loading Bearing Vibration Dataset")
        
        # Placeholder for bearing data loading
        # This would typically load from PRONOSTIA or IMS bearing datasets
        bearing_data = self._simulate_bearing_data()
        return bearing_data
    
    def _download_nasa_data(self):
        """Download NASA CMAPSS dataset if not exists"""
        data_dir = "data/raw/CMAPSSData"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            
            # Download from NASA repository
            url = "https://ti.arc.nasa.gov/c/6/"
            logger.info("Please download NASA CMAPSS data manually from:")
            logger.info("https://ti.arc.nasa.gov/c/6/")
            logger.info("Place files in data/raw/CMAPSSData/ directory")
    
    def _preprocess_nasa_data(self, train_data, test_data, rul_data):
        """Preprocess NASA data for our pipeline format"""
        
        # Convert to our standard format
        sensor_data = []
        maintenance_data = []
        operational_data = []
        
        # Process training data
        for unit_id in train_data['unit'].unique():
            unit_data = train_data[train_data['unit'] == unit_id]
            
            for _, row in unit_data.iterrows():
                sensor_data.append({
                    'timestamp': f"2023-01-01 {row['cycle']:02d}:00:00",
                    'equipment_id': f'Engine_{unit_id:03d}',
                    'vibration': row['sensor_1'] if 'sensor_1' in row else 0,
                    'temperature': row['sensor_2'] if 'sensor_2' in row else 0,
                    'pressure': row['sensor_3'] if 'sensor_3' in row else 0,
                    'current': row['sensor_4'] if 'sensor_4' in row else 0,
                    'rpm': row['sensor_5'] if 'sensor_5' in row else 0,
                    'degradation_level': row['cycle'] / unit_data['cycle'].max(),
                    'failure_imminent': row['cycle'] > unit_data['cycle'].max() * 0.9
                })
        
        # Create maintenance logs based on failure points
        for unit_id in train_data['unit'].unique():
            unit_data = train_data[train_data['unit'] == unit_id]
            max_cycle = unit_data['cycle'].max()
            
            maintenance_data.append({
                'timestamp': f"2023-01-01 {max_cycle:02d}:00:00",
                'equipment_id': f'Engine_{unit_id:03d}',
                'maintenance_type': 'corrective',
                'description': 'Engine failure - replacement required',
                'parts_replaced': 'turbine blades, bearings',
                'technician': 'NASA_Tech_01',
                'downtime_hours': 48
            })
        
        return {
            'sensor': pd.DataFrame(sensor_data),
            'maintenance': pd.DataFrame(maintenance_data),
            'operational': pd.DataFrame(operational_data) if operational_data else pd.DataFrame()
        }
    
    def _preprocess_ai4i_data(self, data):
        """Preprocess AI4I data for our pipeline"""
        
        sensor_data = []
        maintenance_data = []
        
        for _, row in data.iterrows():
            sensor_data.append({
                'timestamp': f"2020-01-{int(row['UDI']//24)+1:02d} {int(row['UDI']%24):02d}:00:00",
                'equipment_id': f"Machine_{row['Product ID'].replace('M', '')}",
                'vibration': row['Air temperature [K]'] / 100,
                'temperature': row['Process temperature [K]'],
                'pressure': row['Rotational speed [rpm]'] / 10,
                'current': row['Torque [Nm]'],
                'rpm': row['Rotational speed [rpm]'],
                'degradation_level': row['Tool wear [min]'] / data['Tool wear [min]'].max(),
                'failure_imminent': row['Machine failure'] == 1
            })
            
            if row['Machine failure'] == 1:
                maintenance_data.append({
                    'timestamp': f"2020-01-{int(row['UDI']//24)+1:02d} {int(row['UDI']%24):02d}:00:00",
                    'equipment_id': f"Machine_{row['Product ID'].replace('M', '')}",
                    'maintenance_type': 'corrective',
                    'description': 'Machine failure detected',
                    'parts_replaced': 'tool, bearings',
                    'technician': 'Tech_01',
                    'downtime_hours': 8
                })
        
        operational_data = pd.DataFrame([{
            'timestamp': '2020-01-01 00:00:00',
            'equipment_id': f"Machine_{i}",
            'production_rate': 95.0,
            'shift': 'A',
            'operator_id': f"OP_{i:03d}"
        } for i in range(1, 100)])
        
        return {
            'sensor': pd.DataFrame(sensor_data),
            'maintenance': pd.DataFrame(maintenance_data),
            'operational': operational_data
        }
    
    def _simulate_bearing_data(self):
        """Simulate bearing vibration data for testing"""
        # This would be replaced with actual bearing dataset loading
        return self.load_ai4i2020()  # Fallback to AI4I for now
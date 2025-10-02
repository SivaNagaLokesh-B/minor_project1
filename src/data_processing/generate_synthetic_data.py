import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def generate_synthetic_data(num_equipment=10, days_of_data=90):
    """Generate realistic synthetic predictive maintenance data"""
    
    logger.info(f"Generating synthetic data for {num_equipment} equipment over {days_of_data} days")
    
    all_data = []
    maintenance_logs = []
    
    for equipment_id in range(1, num_equipment + 1):
        equipment_data = generate_equipment_data(equipment_id, days_of_data)
        equipment_maintenance = generate_maintenance_logs(equipment_data, equipment_id)
        
        all_data.append(equipment_data)
        maintenance_logs.append(equipment_maintenance)
    
    # Combine all equipment data
    sensor_data = pd.concat(all_data, ignore_index=True)
    maintenance_data = pd.concat(maintenance_logs, ignore_index=True)
    
    logger.info(f"Generated {len(sensor_data):,} sensor records and {len(maintenance_data):,} maintenance events")
    
    return sensor_data, maintenance_data

def generate_equipment_data(equipment_id, days_of_data):
    """Generate sensor data for a single equipment with degradation patterns"""
    
    start_date = datetime(2023, 1, 1)
    hours_of_data = days_of_data * 24
    timestamps = [start_date + timedelta(hours=i) for i in range(hours_of_data)]
    
    data = []
    for i, timestamp in enumerate(timestamps):
        # Base values for healthy equipment
        base_vibration = 2.0
        base_temperature = 75.0
        base_pressure = 100.0
        base_current = 15.0
        
        # Introduce degradation over time (non-linear)
        time_progress = i / len(timestamps)
        degradation_factor = min((time_progress / 0.8) ** 2, 1.0)  # Degrade over 80% of timeline
        
        # Add some random noise
        noise_vibration = np.random.normal(0, 0.1)
        noise_temperature = np.random.normal(0, 0.5)
        noise_pressure = np.random.normal(0, 0.3)
        noise_current = np.random.normal(0, 0.2)
        
        # Apply degradation effects
        vibration = base_vibration * (1 + degradation_factor * 2) + noise_vibration
        temperature = base_temperature * (1 + degradation_factor * 0.3) + noise_temperature
        pressure = base_pressure * (1 - degradation_factor * 0.2) + noise_pressure
        current = base_current * (1 + degradation_factor * 0.4) + noise_current
        
        # Failure imminent when degradation is high
        failure_imminent = degradation_factor > 0.85
        
        data.append({
            'timestamp': timestamp,
            'equipment_id': f'EQ_{equipment_id:03d}',
            'vibration': max(0.1, vibration),
            'temperature': max(20, temperature),
            'pressure': max(10, pressure),
            'current': max(5, current),
            'rpm': 1800 + np.random.normal(0, 10),
            'degradation_level': degradation_factor,
            'failure_imminent': failure_imminent
        })
    
    return pd.DataFrame(data)

def generate_maintenance_logs(equipment_data, equipment_id):
    """Generate maintenance logs based on equipment degradation"""
    
    maintenance_events = []
    last_maintenance = equipment_data['timestamp'].min()
    degradation_threshold = 0.8
    
    for _, row in equipment_data.iterrows():
        if (row['degradation_level'] > degradation_threshold and 
            (row['timestamp'] - last_maintenance).total_seconds() > 30 * 24 * 3600):  # Min 30 days between maintenance
            
            maintenance_type = 'corrective' if row['failure_imminent'] else 'preventive'
            downtime = np.random.uniform(4, 48) if maintenance_type == 'corrective' else np.random.uniform(2, 8)
            
            maintenance_events.append({
                'timestamp': row['timestamp'],
                'equipment_id': row['equipment_id'],
                'maintenance_type': maintenance_type,
                'description': f"Replaced bearings and calibrated sensors",
                'parts_replaced': 'bearings, sensors',
                'technician': f"Tech_{np.random.randint(1, 10):02d}",
                'downtime_hours': downtime
            })
            last_maintenance = row['timestamp']
    
    return pd.DataFrame(maintenance_events)

def generate_operational_data(sensor_data):
    """Generate operational context data"""
    
    logger.info("Generating operational data...")
    
    operational_data = []
    unique_equipment = sensor_data['equipment_id'].unique()
    
    for equipment_id in unique_equipment:
        eq_data = sensor_data[sensor_data['equipment_id'] == equipment_id]
        start_time = eq_data['timestamp'].min()
        end_time = eq_data['timestamp'].max()
        
        # Generate daily operational records
        current_time = start_time
        while current_time <= end_time:
            # Operational parameters that change daily
            operational_data.append({
                'timestamp': current_time,
                'equipment_id': equipment_id,
                'production_rate': np.random.uniform(80, 100),  # Percentage
                'shift': np.random.choice(['A', 'B', 'C']),
                'operator_id': f"OP_{np.random.randint(1, 20):03d}",
                'ambient_temperature': np.random.uniform(15, 35),
                'ambient_humidity': np.random.uniform(30, 80),
                'product_type': np.random.choice(['Type_A', 'Type_B', 'Type_C'])
            })
            
            current_time += timedelta(days=1)
    
    operational_df = pd.DataFrame(operational_data)
    logger.info(f"Generated {len(operational_df):,} operational records")
    return operational_df

# Test function
def test_data_generation():
    """Test the data generation functions"""
    print("Testing synthetic data generation...")
    
    sensor_data, maintenance_data = generate_synthetic_data(num_equipment=2, days_of_data=7)
    operational_data = generate_operational_data(sensor_data)
    
    print(f"Sensor data shape: {sensor_data.shape}")
    print(f"Maintenance data shape: {maintenance_data.shape}")
    print(f"Operational data shape: {operational_data.shape}")
    
    print("\nSensor data sample:")
    print(sensor_data.head())
    
    print("\nMaintenance data sample:")
    print(maintenance_data.head())
    
    return sensor_data, maintenance_data, operational_data

if __name__ == "__main__":
    test_data_generation()
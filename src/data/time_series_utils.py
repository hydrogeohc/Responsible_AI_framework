import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import glob
from typing import Tuple, List, Optional

class TimeSeriesWearableDataset(Dataset):
    def __init__(self, sequences, labels, transform=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label

def load_sensor_data(subject_folder: str, activity: str = 'STRESS') -> Optional[pd.DataFrame]:
    """Load sensor data from Empatica E4 files for a specific subject and activity"""
    try:
        activity_path = os.path.join(subject_folder, activity)
        if not os.path.exists(activity_path):
            return None
        
        # Read sensor files
        sensor_data = {}
        
        # EDA (Electrodermal Activity) - 4 Hz
        eda_file = os.path.join(activity_path, 'EDA.csv')
        if os.path.exists(eda_file):
            eda_df = pd.read_csv(eda_file, header=None)
            start_time = float(eda_df.iloc[0, 0])
            sample_rate = float(eda_df.iloc[1, 0])
            eda_values = eda_df.iloc[2:, 0].values.astype(float)
            sensor_data['EDA'] = eda_values
        
        # Heart Rate - 1 Hz
        hr_file = os.path.join(activity_path, 'HR.csv')
        if os.path.exists(hr_file):
            hr_df = pd.read_csv(hr_file, header=None)
            hr_values = hr_df.iloc[2:, 0].values.astype(float)
            sensor_data['HR'] = hr_values
        
        # Temperature - 4 Hz
        temp_file = os.path.join(activity_path, 'TEMP.csv')
        if os.path.exists(temp_file):
            temp_df = pd.read_csv(temp_file, header=None)
            temp_values = temp_df.iloc[2:, 0].values.astype(float)
            sensor_data['TEMP'] = temp_values
        
        # Accelerometer - 32 Hz (we'll downsample)
        acc_file = os.path.join(activity_path, 'ACC.csv')
        if os.path.exists(acc_file):
            acc_df = pd.read_csv(acc_file, header=None)
            acc_x = acc_df.iloc[2:, 0].values.astype(float)
            acc_y = acc_df.iloc[2:, 1].values.astype(float)
            acc_z = acc_df.iloc[2:, 2].values.astype(float)
            # Calculate magnitude
            acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            # Downsample to 4 Hz
            acc_magnitude = acc_magnitude[::8]  # 32/4 = 8
            sensor_data['ACC'] = acc_magnitude
        
        # Align all sensors to the same length (minimum length)
        min_length = min(len(data) for data in sensor_data.values())
        
        # Create time series dataframe
        df = pd.DataFrame({
            'EDA': sensor_data.get('EDA', [0]*min_length)[:min_length],
            'HR': np.repeat(sensor_data.get('HR', [0]*min_length), 4)[:min_length],  # Upsample HR
            'TEMP': sensor_data.get('TEMP', [0]*min_length)[:min_length],
            'ACC': sensor_data.get('ACC', [0]*min_length)[:min_length]
        })
        
        return df
        
    except Exception as e:
        print(f"Error loading data for {subject_folder}: {e}")
        return None

def create_time_series_sequences(data: pd.DataFrame, stress_level: int, 
                                sequence_length: int = 60, stride: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Create time series sequences from sensor data"""
    sequences = []
    labels = []
    
    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    for i in range(0, len(data_scaled) - sequence_length + 1, stride):
        sequence = data_scaled[i:i + sequence_length]
        sequences.append(sequence)
        labels.append(stress_level)
    
    return np.array(sequences), np.array(labels)

def load_time_series_stress_data(dataset_path: str = "wearable_dataset", 
                                sequence_length: int = 60, stride: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Load time series stress data from wearable dataset"""
    
    # Read stress level data
    stress_v1_path = os.path.join(dataset_path, "Stress_Level_v1.csv")
    stress_v2_path = os.path.join(dataset_path, "Stress_Level_v2.csv")
    
    if not os.path.exists(stress_v1_path):
        print("Stress level files not found, generating synthetic time series data...")
        return generate_synthetic_time_series(sequence_length=sequence_length)
    
    stress_v1 = pd.read_csv(stress_v1_path)
    stress_v2 = pd.read_csv(stress_v2_path)
    
    all_sequences = []
    all_labels = []
    
    # Process each subject
    for idx, row in stress_v1.iterrows():
        subject_id = row.iloc[0]
        
        # Get average stress level for this subject
        stress_values = row.iloc[1:].values
        avg_stress = np.nanmean(stress_values[~pd.isna(stress_values)])
        
        # Categorize stress level
        if avg_stress <= 3:
            stress_category = 0  # Low
        elif avg_stress <= 6:
            stress_category = 1  # Medium
        else:
            stress_category = 2  # High
        
        # Load sensor data for this subject
        subject_folder = os.path.join(dataset_path, subject_id)
        if os.path.exists(subject_folder):
            sensor_data = load_sensor_data(subject_folder, 'STRESS')
            if sensor_data is not None and len(sensor_data) > sequence_length:
                sequences, labels = create_time_series_sequences(
                    sensor_data, stress_category, sequence_length, stride
                )
                all_sequences.extend(sequences)
                all_labels.extend(labels)
    
    if len(all_sequences) == 0:
        print("No valid sensor data found, generating synthetic time series data...")
        return generate_synthetic_time_series(sequence_length=sequence_length)
    
    return np.array(all_sequences), np.array(all_labels)

def generate_synthetic_time_series(n_samples: int = 1000, sequence_length: int = 60, 
                                  n_features: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time series data for demonstration"""
    np.random.seed(42)
    
    sequences = []
    labels = []
    
    for i in range(n_samples):
        # Generate stress level
        stress_level = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        
        # Generate base patterns for each stress level
        if stress_level == 0:  # Low stress
            base_eda = np.random.normal(2.0, 0.5, sequence_length)
            base_hr = np.random.normal(70, 5, sequence_length)
            base_temp = np.random.normal(32.0, 0.5, sequence_length)
            base_acc = np.random.normal(0.5, 0.2, sequence_length)
        elif stress_level == 1:  # Medium stress
            base_eda = np.random.normal(4.0, 0.8, sequence_length)
            base_hr = np.random.normal(85, 8, sequence_length)
            base_temp = np.random.normal(32.5, 0.7, sequence_length)
            base_acc = np.random.normal(0.8, 0.3, sequence_length)
        else:  # High stress
            base_eda = np.random.normal(6.0, 1.2, sequence_length)
            base_hr = np.random.normal(100, 12, sequence_length)
            base_temp = np.random.normal(33.0, 1.0, sequence_length)
            base_acc = np.random.normal(1.2, 0.5, sequence_length)
        
        # Add temporal patterns
        t = np.linspace(0, 1, sequence_length)
        
        # Add some trend and seasonality
        trend = np.sin(2 * np.pi * t) * 0.5
        base_eda += trend * (stress_level + 1)
        base_hr += trend * (stress_level + 1) * 5
        base_temp += trend * (stress_level + 1) * 0.2
        base_acc += trend * (stress_level + 1) * 0.1
        
        # Stack features
        sequence = np.column_stack([base_eda, base_hr, base_temp, base_acc])
        sequences.append(sequence)
        labels.append(stress_level)
    
    return np.array(sequences), np.array(labels)

def create_time_series_data_loaders(sequences: np.ndarray, labels: np.ndarray, 
                                   batch_size: int = 32, test_size: float = 0.2, 
                                   val_size: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders for time series data"""
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = TimeSeriesWearableDataset(X_train, y_train)
    val_dataset = TimeSeriesWearableDataset(X_val, y_val)
    test_dataset = TimeSeriesWearableDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_sample_time_series_for_demo(sequence_length: int = 60) -> Tuple[np.ndarray, int]:
    """Generate a sample time series for demonstration"""
    np.random.seed(42)
    
    # Generate a medium stress example
    stress_level = 1
    
    # Generate realistic physiological patterns
    t = np.linspace(0, 1, sequence_length)
    
    # EDA (microsiemens) - gradual increase under stress
    eda = 3.0 + 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.2, sequence_length)
    
    # Heart Rate (BPM) - elevated and variable under stress
    hr = 80 + 10 * np.sin(4 * np.pi * t) + np.random.normal(0, 3, sequence_length)
    
    # Temperature (Celsius) - slight increase
    temp = 32.5 + 0.3 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, sequence_length)
    
    # Accelerometer (g) - increased movement
    acc = 0.8 + 0.2 * np.sin(6 * np.pi * t) + np.random.normal(0, 0.1, sequence_length)
    
    # Stack features
    sequence = np.column_stack([eda, hr, temp, acc])
    
    return sequence, stress_level
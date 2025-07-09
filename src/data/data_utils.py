import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

class WearableDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_stress_data(dataset_path="wearable_dataset"):
    stress_v1 = pd.read_csv(os.path.join(dataset_path, "Stress_Level_v1.csv"))
    stress_v2 = pd.read_csv(os.path.join(dataset_path, "Stress_Level_v2.csv"))
    subject_info = pd.read_csv(os.path.join(dataset_path, "subject-info.csv"))
    
    # Combine stress data
    stress_data = pd.concat([stress_v1, stress_v2], ignore_index=True)
    
    # Extract features from subject info
    features = []
    labels = []
    
    for idx, row in stress_data.iterrows():
        subject_id = row.iloc[0]  # First column is subject ID
        
        # Find subject info
        subject_row = subject_info[subject_info['Info '] == subject_id]
        if not subject_row.empty:
            # Extract physiological features (Age, Height, Weight, Activity)
            age = subject_row['Age'].values[0]
            height = subject_row['Height (cm)'].values[0]
            weight = subject_row['Weight (kg)'].values[0]
            activity = 1 if subject_row['Does physical activity regularly?'].values[0] == 'Yes' else 0
            
            # Extract stress levels for different activities
            stress_levels = row.iloc[1:].values  # Skip subject ID
            
            # Create feature vector for each stress measurement
            for stress_level in stress_levels:
                if not pd.isna(stress_level):
                    features.append([age, height, weight, activity])
                    # Categorize stress levels: Low (0-3), Medium (4-6), High (7-10)
                    if stress_level <= 3:
                        labels.append(0)
                    elif stress_level <= 6:
                        labels.append(1)
                    else:
                        labels.append(2)
    
    return np.array(features), np.array(labels)

def create_federated_data_loaders(features, labels, num_clients=3, batch_size=32):
    # Split data among clients
    client_data = []
    client_labels = []
    
    # Stratified split to ensure each client has samples from all classes
    for i in range(num_clients):
        start_idx = i * len(features) // num_clients
        end_idx = (i + 1) * len(features) // num_clients
        
        client_features = features[start_idx:end_idx]
        client_label = labels[start_idx:end_idx]
        
        # Normalize features
        scaler = StandardScaler()
        client_features_scaled = scaler.fit_transform(client_features)
        
        # Create dataset and dataloader
        dataset = WearableDataset(client_features_scaled, client_label)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        client_data.append(dataloader)
        client_labels.append(client_label)
    
    return client_data, client_labels

def get_sample_data_for_demo():
    """Generate sample data for demonstration purposes"""
    np.random.seed(42)
    
    # Generate synthetic physiological data
    n_samples = 100
    features = []
    labels = []
    
    for i in range(n_samples):
        # Age (18-65), Height (150-200), Weight (50-120), Activity (0/1)
        age = np.random.randint(18, 66)
        height = np.random.randint(150, 201)
        weight = np.random.randint(50, 121)
        activity = np.random.choice([0, 1])
        
        # Simulate stress level based on features
        stress_score = (
            (age - 40) * 0.1 +
            (weight - 70) * 0.05 +
            activity * (-1) +
            np.random.normal(0, 1)
        )
        
        features.append([age, height, weight, activity])
        
        # Categorize stress
        if stress_score <= -0.5:
            labels.append(0)  # Low stress
        elif stress_score <= 0.5:
            labels.append(1)  # Medium stress
        else:
            labels.append(2)  # High stress
    
    return np.array(features), np.array(labels)
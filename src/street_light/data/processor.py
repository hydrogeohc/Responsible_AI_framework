"""
Street Light Data Processor
Handles data preparation, feature engineering, and integration with the responsible AI framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class StreetLightDataProcessor:
    """
    Data processor for street light IoT data integration.
    
    Handles data preparation, feature engineering, and integration
    with the existing responsible AI framework.
    """
    
    def __init__(self):
        """Initialize data processor."""
        self.feature_columns = [
            'led_count', 'operational_percentage', 'repair_rate',
            'energy_consumption', 'age_years', 'location_density',
            'weather_factor', 'usage_pattern'
        ]
        
    def load_street_light_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess street light data.
        
        Args:
            data_path: Path to street light data CSV
            
        Returns:
            Processed DataFrame
        """
        try:
            df = pd.read_csv(data_path)
            
            # Parse dates
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            if 'Date Value' in df.columns:
                df['Date Value'] = pd.to_datetime(df['Date Value'])
                
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Engineer features for model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Feature array for model training
        """
        features = []
        
        if 'Cumulative # of streetlights converted to LED' in df.columns:
            # LED conversion features
            led_counts = df['Cumulative # of streetlights converted to LED'].fillna(0)
            features.append(led_counts.values)
            
        if '% outages repaired within 10 business days' in df.columns:
            # Repair efficiency features
            repair_rates = df['% outages repaired within 10 business days'].fillna(90)
            features.append(repair_rates.values)
            
        # Time-based features
        if 'Date' in df.columns:
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            features.extend([df['year'].values, df['month'].values])
            
        # Synthetic features for demonstration
        n_samples = len(df)
        features.extend([
            np.random.normal(5, 2, n_samples),  # Age in years
            np.random.uniform(0.5, 1.5, n_samples),  # Location density
            np.random.uniform(0.8, 1.2, n_samples),  # Weather factor
            np.random.uniform(0.6, 1.4, n_samples),  # Usage pattern
        ])
        
        # Ensure we have the right number of features
        while len(features) < 8:
            features.append(np.ones(n_samples))
            
        return np.column_stack(features[:8])
    
    def create_time_series_data(self, df: pd.DataFrame, 
                               sequence_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series data for LED conversion model.
        
        Args:
            df: Input DataFrame
            sequence_length: Length of time series sequences
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        features = self.engineer_features(df)
        
        if len(features) < sequence_length:
            # Pad with repeated values if insufficient data
            padding = np.repeat(features[-1:], sequence_length - len(features), axis=0)
            features = np.vstack([features, padding])
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            # Target: next LED conversion count and estimated energy savings
            led_count = features[i+sequence_length, 0] if len(features[i+sequence_length]) > 0 else 0
            energy_savings = led_count * 0.5  # Estimate 50% energy savings per LED
            y.append([led_count, energy_savings])
        
        return np.array(X), np.array(y)
    
    def calculate_carbon_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate carbon footprint features for street lights.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Carbon-related features
        """
        base_features = self.engineer_features(df)
        
        # Additional carbon-specific features
        n_samples = len(df)
        carbon_features = [
            np.random.uniform(100, 200, n_samples),  # Base energy consumption
            np.random.uniform(0.4, 0.6, n_samples),  # Carbon intensity factor
            np.random.uniform(0.7, 1.3, n_samples),  # Grid efficiency
            np.random.uniform(0.1, 0.3, n_samples),  # Renewable energy %
        ]
        
        # Combine with base features
        all_features = np.column_stack([base_features, *carbon_features])
        
        return all_features[:, :10]  # Take first 10 features
    
    def get_sample_data(self, data_path: str = None, n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Get sample data for model testing.
        
        Args:
            data_path: Path to street light data
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with sample data for different models
        """
        if data_path:
            df = self.load_street_light_data(data_path)
        else:
            df = pd.DataFrame()
        
        if df.empty:
            # Generate synthetic data
            df = pd.DataFrame({
                'Date': pd.date_range('2020-01-01', periods=n_samples, freq='M'),
                'Cumulative # of streetlights converted to LED': np.random.randint(1000, 200000, n_samples),
                '% outages repaired within 10 business days': np.random.randint(75, 99, n_samples)
            })
        
        # Ensure we have enough samples
        if len(df) < n_samples:
            # Repeat data to reach required samples
            repeat_factor = (n_samples // len(df)) + 1
            df = pd.concat([df] * repeat_factor).reset_index(drop=True)
            df = df.head(n_samples)
        
        performance_features = self.engineer_features(df)
        carbon_features = self.calculate_carbon_features(df)
        time_series_X, time_series_y = self.create_time_series_data(df)
        
        # Create synthetic targets
        performance_targets = np.random.randint(0, 3, len(performance_features))  # 0=good, 1=maintenance, 2=failure
        carbon_targets = np.random.uniform(0.1, 2.0, len(carbon_features))  # CO2 kg per light
        
        return {
            'performance': {
                'features': performance_features,
                'targets': performance_targets
            },
            'carbon': {
                'features': carbon_features,
                'targets': carbon_targets
            },
            'time_series': {
                'features': time_series_X,
                'targets': time_series_y
            }
        }
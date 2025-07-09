"""
Street Light Data Loader
Handles loading and validation of street light data from various sources.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Union
from pathlib import Path


class StreetLightDataLoader:
    """
    Data loader for street light datasets.
    
    Handles loading data from CSV files, validation, and basic preprocessing.
    """
    
    def __init__(self, data_dir: str = "smart_city_light"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing street light data files
        """
        self.data_dir = Path(data_dir)
        self.supported_files = {
            'led_conversion': 'bsl-cumulative-number-of-streetlights-converted-to-led.csv',
            'outage_repair': 'bsl-percent-of-streetlight-outages-repaired-within-10-business-days.csv',
            'performance_metrics': 'bureau-of-street-lighting-bsl-performance-metrics.csv'
        }
        
    def load_led_conversion_data(self) -> pd.DataFrame:
        """
        Load LED conversion data.
        
        Returns:
            DataFrame with LED conversion data
        """
        file_path = self.data_dir / self.supported_files['led_conversion']
        
        if not file_path.exists():
            raise FileNotFoundError(f"LED conversion data not found at {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Parse dates
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Validate required columns
        required_cols = ['Date Name', 'Date', 'Cumulative # of streetlights converted to LED']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def load_outage_repair_data(self) -> pd.DataFrame:
        """
        Load outage repair data.
        
        Returns:
            DataFrame with outage repair data
        """
        file_path = self.data_dir / self.supported_files['outage_repair']
        
        if not file_path.exists():
            raise FileNotFoundError(f"Outage repair data not found at {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Parse dates
        if 'Date Value' in df.columns:
            df['Date Value'] = pd.to_datetime(df['Date Value'])
        
        # Validate required columns
        required_cols = ['Date Name', 'Date Value', '% outages repaired within 10 business days']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def load_performance_metrics_data(self) -> pd.DataFrame:
        """
        Load performance metrics data.
        
        Returns:
            DataFrame with performance metrics data
        """
        file_path = self.data_dir / self.supported_files['performance_metrics']
        
        if not file_path.exists():
            raise FileNotFoundError(f"Performance metrics data not found at {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Parse dates
        if 'Date Value' in df.columns:
            df['Date Value'] = pd.to_datetime(df['Date Value'])
        
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available street light data.
        
        Returns:
            Dictionary with all loaded datasets
        """
        data = {}
        
        # Try to load each dataset
        for key, filename in self.supported_files.items():
            try:
                if key == 'led_conversion':
                    data[key] = self.load_led_conversion_data()
                elif key == 'outage_repair':
                    data[key] = self.load_outage_repair_data()
                elif key == 'performance_metrics':
                    data[key] = self.load_performance_metrics_data()
                    
                print(f"✓ Loaded {key} data: {len(data[key])} records")
                
            except Exception as e:
                print(f"⚠️  Could not load {key} data: {e}")
                data[key] = pd.DataFrame()
        
        return data
    
    def validate_data(self, df: pd.DataFrame, data_type: str) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate street light data.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data ('led_conversion', 'outage_repair', 'performance_metrics')
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if DataFrame is empty
        if df.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("DataFrame is empty")
            return validation_result
        
        # Data-specific validation
        if data_type == 'led_conversion':
            # Check for required columns
            required_cols = ['Date Name', 'Date', 'Cumulative # of streetlights converted to LED']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing columns: {missing_cols}")
            
            # Check for negative LED counts
            if 'Cumulative # of streetlights converted to LED' in df.columns:
                negative_counts = df['Cumulative # of streetlights converted to LED'] < 0
                if negative_counts.any():
                    validation_result['warnings'].append("Found negative LED counts")
        
        elif data_type == 'outage_repair':
            # Check for required columns
            required_cols = ['Date Name', 'Date Value', '% outages repaired within 10 business days']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing columns: {missing_cols}")
            
            # Check for invalid percentages
            if '% outages repaired within 10 business days' in df.columns:
                invalid_percentages = (df['% outages repaired within 10 business days'] < 0) | \
                                    (df['% outages repaired within 10 business days'] > 100)
                if invalid_percentages.any():
                    validation_result['warnings'].append("Found invalid percentage values")
        
        # General validation
        # Check for duplicate dates
        date_columns = [col for col in df.columns if 'Date' in col and df[col].dtype == 'datetime64[ns]']
        for date_col in date_columns:
            if df[date_col].duplicated().any():
                validation_result['warnings'].append(f"Found duplicate dates in {date_col}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            validation_result['warnings'].append(f"Found missing values: {missing_values[missing_values > 0].to_dict()}")
        
        return validation_result
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for street light data.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {'error': 'DataFrame is empty'}
        
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        # Add date range if date columns exist
        date_columns = [col for col in df.columns if 'Date' in col and df[col].dtype == 'datetime64[ns]']
        if date_columns:
            summary['date_ranges'] = {}
            for date_col in date_columns:
                summary['date_ranges'][date_col] = {
                    'start': df[date_col].min().strftime('%Y-%m-%d'),
                    'end': df[date_col].max().strftime('%Y-%m-%d'),
                    'span_days': (df[date_col].max() - df[date_col].min()).days
                }
        
        return summary
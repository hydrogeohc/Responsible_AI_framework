"""
Street Light Helper Functions
Utility functions for street light data processing and analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta


def create_sample_district_data(n_districts: int = 3, samples_per_district: int = 100) -> List[pd.DataFrame]:
    """
    Create sample data for multiple city districts.
    
    Args:
        n_districts: Number of districts to create
        samples_per_district: Number of samples per district
        
    Returns:
        List of DataFrames for each district
    """
    district_data = []
    
    for i in range(n_districts):
        # Create district-specific data with some variation
        base_led_count = 10000 + i * 5000
        base_operational = 95 + i * 2
        
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=samples_per_district, freq='D'),
            'Cumulative # of streetlights converted to LED': np.random.randint(
                base_led_count, base_led_count + 50000, samples_per_district
            ),
            '% outages repaired within 10 business days': np.random.randint(
                base_operational - 10, base_operational + 5, samples_per_district
            ),
            'District': f'District_{i+1}'
        })
        
        district_data.append(df)
    
    return district_data


def calculate_energy_savings(traditional_lights: int, led_lights: int, 
                           traditional_power: float = 150.0, led_power: float = 75.0,
                           hours_per_day: float = 12.0) -> Dict[str, float]:
    """
    Calculate energy savings from LED conversion.
    
    Args:
        traditional_lights: Number of traditional lights
        led_lights: Number of LED lights
        traditional_power: Power consumption of traditional lights (watts)
        led_power: Power consumption of LED lights (watts)
        hours_per_day: Operating hours per day
        
    Returns:
        Dictionary with energy savings metrics
    """
    # Daily energy consumption (kWh)
    traditional_daily = (traditional_lights * traditional_power * hours_per_day) / 1000
    led_daily = (led_lights * led_power * hours_per_day) / 1000
    
    # Annual energy consumption (kWh)
    traditional_annual = traditional_daily * 365
    led_annual = led_daily * 365
    
    # Energy savings
    daily_savings = traditional_daily - led_daily
    annual_savings = traditional_annual - led_annual
    
    # Cost savings (assuming $0.12 per kWh)
    cost_per_kwh = 0.12
    annual_cost_savings = annual_savings * cost_per_kwh
    
    # Carbon savings (assuming 0.5 kg CO2 per kWh)
    carbon_intensity = 0.5
    annual_carbon_savings = annual_savings * carbon_intensity
    
    return {
        'daily_energy_savings_kwh': daily_savings,
        'annual_energy_savings_kwh': annual_savings,
        'annual_cost_savings_usd': annual_cost_savings,
        'annual_carbon_savings_kg': annual_carbon_savings,
        'savings_percentage': (annual_savings / traditional_annual) * 100 if traditional_annual > 0 else 0
    }


def generate_synthetic_weather_data(start_date: str, end_date: str, 
                                  location: str = "city_center") -> pd.DataFrame:
    """
    Generate synthetic weather data for street light analysis.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        location: Location identifier
        
    Returns:
        DataFrame with synthetic weather data
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic weather data
    np.random.seed(42)  # For reproducibility
    
    weather_data = pd.DataFrame({
        'date': date_range,
        'temperature_celsius': np.random.normal(20, 10, len(date_range)),
        'humidity_percent': np.random.uniform(30, 90, len(date_range)),
        'wind_speed_kmh': np.random.exponential(10, len(date_range)),
        'precipitation_mm': np.random.exponential(2, len(date_range)),
        'cloud_cover_percent': np.random.uniform(0, 100, len(date_range)),
        'visibility_km': np.random.uniform(5, 50, len(date_range)),
        'location': location
    })
    
    return weather_data


def calculate_maintenance_priority(led_count: int, operational_percentage: float,
                                 repair_rate: float, age_years: float) -> Dict[str, float]:
    """
    Calculate maintenance priority scores for street lights.
    
    Args:
        led_count: Number of LED lights
        operational_percentage: Percentage of lights operational
        repair_rate: Percentage of repairs completed on time
        age_years: Average age of lights in years
        
    Returns:
        Dictionary with priority scores
    """
    # Normalize inputs
    operational_score = max(0, min(1, operational_percentage / 100))
    repair_score = max(0, min(1, repair_rate / 100))
    age_score = max(0, min(1, (10 - age_years) / 10))  # Assuming 10 years max age
    
    # Calculate priority components
    performance_priority = (1 - operational_score) * 0.4
    maintenance_priority = (1 - repair_score) * 0.3
    aging_priority = (1 - age_score) * 0.3
    
    # Overall priority
    overall_priority = performance_priority + maintenance_priority + aging_priority
    
    # Priority categories
    if overall_priority < 0.3:
        priority_category = "Low"
    elif overall_priority < 0.6:
        priority_category = "Medium"
    else:
        priority_category = "High"
    
    return {
        'performance_priority': performance_priority,
        'maintenance_priority': maintenance_priority,
        'aging_priority': aging_priority,
        'overall_priority': overall_priority,
        'priority_category': priority_category
    }


def format_prediction_output(prediction: Dict, model_type: str) -> str:
    """
    Format model predictions for display.
    
    Args:
        prediction: Dictionary with prediction results
        model_type: Type of model ('performance', 'carbon', 'led')
        
    Returns:
        Formatted string output
    """
    if model_type == "performance":
        status = prediction.get('predicted_status', 'Unknown')
        confidence = prediction.get('confidence', 0)
        energy = prediction.get('energy_consumption_kwh', 0)
        carbon = prediction.get('carbon_footprint_kg', 0)
        
        return f"""
Street Light Performance Prediction:
├── Status: {status}
├── Confidence: {confidence:.1%}
├── Energy Consumption: {energy:.2f} kWh
└── Carbon Footprint: {carbon:.4f} kg CO2
"""
    
    elif model_type == "carbon":
        carbon_level = prediction.get('carbon_level', 'Unknown')
        carbon_kg = prediction.get('predicted_carbon_kg', 0)
        
        return f"""
Carbon Footprint Prediction:
├── Carbon Level: {carbon_level}
└── Predicted CO2: {carbon_kg:.4f} kg
"""
    
    elif model_type == "led":
        conversion_rate = prediction.get('conversion_rate', 0)
        energy_savings = prediction.get('energy_savings', 0)
        
        return f"""
LED Conversion Prediction:
├── Conversion Rate: {conversion_rate:.1%}
└── Energy Savings: {energy_savings:.2f} kWh
"""
    
    else:
        return f"Unknown model type: {model_type}"


def validate_street_light_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate street light configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Required fields
    required_fields = ['privacy_epsilon', 'carbon_log_dir', 'model_params']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate privacy epsilon
    if 'privacy_epsilon' in config:
        epsilon = config['privacy_epsilon']
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            errors.append("privacy_epsilon must be a positive number")
    
    # Validate model parameters
    if 'model_params' in config:
        model_params = config['model_params']
        if not isinstance(model_params, dict):
            errors.append("model_params must be a dictionary")
        else:
            # Check for required model parameters
            required_model_params = ['input_dim', 'hidden_dim']
            for param in required_model_params:
                if param not in model_params:
                    errors.append(f"Missing model parameter: {param}")
    
    return len(errors) == 0, errors
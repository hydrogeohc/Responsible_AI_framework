"""
Street Light Metrics
Utility classes for calculating and tracking street light performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class StreetLightMetrics:
    """
    Utility class for calculating street light performance metrics.
    
    Provides methods for calculating energy savings, carbon footprint,
    operational efficiency, and maintenance metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.carbon_intensity = 0.5  # kg CO2 per kWh
        self.electricity_cost = 0.12  # USD per kWh
        self.traditional_power = 150.0  # watts
        self.led_power = 75.0  # watts
        self.operating_hours = 12.0  # hours per day
        
    def calculate_energy_consumption(self, light_count: int, 
                                   light_type: str = "led", 
                                   days: int = 365) -> Dict[str, float]:
        """
        Calculate energy consumption for street lights.
        
        Args:
            light_count: Number of lights
            light_type: Type of lights ('led' or 'traditional')
            days: Number of days to calculate for
            
        Returns:
            Dictionary with energy consumption metrics
        """
        power_per_light = self.led_power if light_type == "led" else self.traditional_power
        
        # Daily consumption (kWh)
        daily_kwh = (light_count * power_per_light * self.operating_hours) / 1000
        
        # Total consumption for specified days
        total_kwh = daily_kwh * days
        
        # Cost calculation
        total_cost = total_kwh * self.electricity_cost
        
        # Carbon emissions
        carbon_emissions = total_kwh * self.carbon_intensity
        
        return {
            'daily_kwh': daily_kwh,
            'total_kwh': total_kwh,
            'total_cost_usd': total_cost,
            'carbon_emissions_kg': carbon_emissions,
            'lights_count': light_count,
            'light_type': light_type,
            'days': days
        }
    
    def calculate_led_conversion_impact(self, traditional_count: int, 
                                      led_count: int,
                                      days: int = 365) -> Dict[str, float]:
        """
        Calculate impact of LED conversion.
        
        Args:
            traditional_count: Number of traditional lights
            led_count: Number of LED lights
            days: Number of days to calculate for
            
        Returns:
            Dictionary with conversion impact metrics
        """
        # Calculate consumption for both types
        traditional_metrics = self.calculate_energy_consumption(
            traditional_count, "traditional", days
        )
        led_metrics = self.calculate_energy_consumption(
            led_count, "led", days
        )
        
        # Calculate savings
        energy_savings = traditional_metrics['total_kwh'] - led_metrics['total_kwh']
        cost_savings = traditional_metrics['total_cost_usd'] - led_metrics['total_cost_usd']
        carbon_savings = traditional_metrics['carbon_emissions_kg'] - led_metrics['carbon_emissions_kg']
        
        # Calculate percentages
        energy_savings_pct = (energy_savings / traditional_metrics['total_kwh']) * 100 if traditional_metrics['total_kwh'] > 0 else 0
        cost_savings_pct = (cost_savings / traditional_metrics['total_cost_usd']) * 100 if traditional_metrics['total_cost_usd'] > 0 else 0
        carbon_savings_pct = (carbon_savings / traditional_metrics['carbon_emissions_kg']) * 100 if traditional_metrics['carbon_emissions_kg'] > 0 else 0
        
        return {
            'energy_savings_kwh': energy_savings,
            'cost_savings_usd': cost_savings,
            'carbon_savings_kg': carbon_savings,
            'energy_savings_percentage': energy_savings_pct,
            'cost_savings_percentage': cost_savings_pct,
            'carbon_savings_percentage': carbon_savings_pct,
            'total_lights': traditional_count + led_count,
            'led_conversion_rate': (led_count / (traditional_count + led_count)) * 100 if (traditional_count + led_count) > 0 else 0
        }
    
    def calculate_operational_efficiency(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate operational efficiency metrics from street light data.
        
        Args:
            df: DataFrame with street light operational data
            
        Returns:
            Dictionary with operational efficiency metrics
        """
        metrics = {}
        
        # Average operational percentage
        if '% outages repaired within 10 business days' in df.columns:
            repair_rates = df['% outages repaired within 10 business days'].dropna()
            if len(repair_rates) > 0:
                metrics['avg_repair_rate'] = repair_rates.mean()
                metrics['min_repair_rate'] = repair_rates.min()
                metrics['max_repair_rate'] = repair_rates.max()
                metrics['repair_rate_std'] = repair_rates.std()
        
        # LED conversion progress
        if 'Cumulative # of streetlights converted to LED' in df.columns:
            led_counts = df['Cumulative # of streetlights converted to LED'].dropna()
            if len(led_counts) > 0:
                metrics['current_led_count'] = led_counts.iloc[-1]
                metrics['led_growth_rate'] = (led_counts.iloc[-1] - led_counts.iloc[0]) / len(led_counts) if len(led_counts) > 1 else 0
                metrics['led_conversion_velocity'] = led_counts.diff().mean() if len(led_counts) > 1 else 0
        
        # Time-based metrics
        if 'Date' in df.columns or 'Date Value' in df.columns:
            date_col = 'Date' if 'Date' in df.columns else 'Date Value'
            if not df[date_col].empty:
                date_range = df[date_col].max() - df[date_col].min()
                metrics['data_span_days'] = date_range.days
                metrics['data_points'] = len(df)
                metrics['data_density'] = len(df) / max(date_range.days, 1)
        
        return metrics
    
    def calculate_maintenance_metrics(self, operational_pct: float, 
                                    repair_rate: float,
                                    age_years: float) -> Dict[str, float]:
        """
        Calculate maintenance priority and metrics.
        
        Args:
            operational_pct: Percentage of lights operational
            repair_rate: Percentage of repairs completed on time
            age_years: Average age of lights in years
            
        Returns:
            Dictionary with maintenance metrics
        """
        # Normalize inputs (0-1 scale)
        operational_score = max(0, min(1, operational_pct / 100))
        repair_score = max(0, min(1, repair_rate / 100))
        age_score = max(0, min(1, (15 - age_years) / 15))  # Assuming 15 years max useful life
        
        # Calculate priority components
        performance_priority = (1 - operational_score) * 0.4
        maintenance_priority = (1 - repair_score) * 0.3
        aging_priority = (1 - age_score) * 0.3
        
        # Overall priority
        overall_priority = performance_priority + maintenance_priority + aging_priority
        
        # Priority level
        if overall_priority < 0.3:
            priority_level = "Low"
            priority_numeric = 1
        elif overall_priority < 0.6:
            priority_level = "Medium"
            priority_numeric = 2
        else:
            priority_level = "High"
            priority_numeric = 3
        
        # Maintenance recommendations
        recommendations = []
        if operational_score < 0.9:
            recommendations.append("Increase routine maintenance frequency")
        if repair_score < 0.8:
            recommendations.append("Improve repair response times")
        if age_score < 0.5:
            recommendations.append("Consider equipment replacement")
        
        return {
            'operational_score': operational_score,
            'repair_score': repair_score,
            'age_score': age_score,
            'performance_priority': performance_priority,
            'maintenance_priority': maintenance_priority,
            'aging_priority': aging_priority,
            'overall_priority': overall_priority,
            'priority_level': priority_level,
            'priority_numeric': priority_numeric,
            'recommendations': recommendations
        }
    
    def calculate_carbon_footprint(self, energy_kwh: float, 
                                 carbon_intensity: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate carbon footprint from energy consumption.
        
        Args:
            energy_kwh: Energy consumption in kWh
            carbon_intensity: Carbon intensity in kg CO2/kWh (optional)
            
        Returns:
            Dictionary with carbon footprint metrics
        """
        if carbon_intensity is None:
            carbon_intensity = self.carbon_intensity
        
        # Calculate carbon emissions
        carbon_emissions = energy_kwh * carbon_intensity
        
        # Carbon level classification
        if carbon_emissions < 100:
            carbon_level = "Low"
        elif carbon_emissions < 500:
            carbon_level = "Medium"
        else:
            carbon_level = "High"
        
        # Equivalent metrics
        trees_needed = carbon_emissions / 22  # Assuming 22 kg CO2 absorbed per tree per year
        car_miles_equivalent = carbon_emissions / 0.4  # Assuming 0.4 kg CO2 per mile
        
        return {
            'carbon_emissions_kg': carbon_emissions,
            'carbon_level': carbon_level,
            'trees_needed_equivalent': trees_needed,
            'car_miles_equivalent': car_miles_equivalent,
            'carbon_intensity_used': carbon_intensity,
            'energy_kwh': energy_kwh
        }
    
    def generate_sustainability_report(self, led_count: int, 
                                     traditional_count: int,
                                     operational_pct: float,
                                     repair_rate: float) -> Dict:
        """
        Generate comprehensive sustainability report.
        
        Args:
            led_count: Number of LED lights
            traditional_count: Number of traditional lights
            operational_pct: Percentage of lights operational
            repair_rate: Percentage of repairs completed on time
            
        Returns:
            Dictionary with sustainability report
        """
        # Calculate LED conversion impact
        conversion_impact = self.calculate_led_conversion_impact(
            traditional_count, led_count
        )
        
        # Calculate operational efficiency
        operational_metrics = {
            'operational_percentage': operational_pct,
            'repair_rate': repair_rate,
            'total_lights': led_count + traditional_count,
            'led_percentage': (led_count / (led_count + traditional_count)) * 100 if (led_count + traditional_count) > 0 else 0
        }
        
        # Calculate carbon footprint
        total_energy = self.calculate_energy_consumption(led_count, "led")['total_kwh'] + \
                      self.calculate_energy_consumption(traditional_count, "traditional")['total_kwh']
        
        carbon_metrics = self.calculate_carbon_footprint(total_energy)
        
        # Generate recommendations
        recommendations = []
        if operational_metrics['led_percentage'] < 80:
            recommendations.append("Accelerate LED conversion to improve energy efficiency")
        if operational_pct < 95:
            recommendations.append("Improve operational maintenance to reduce outages")
        if repair_rate < 90:
            recommendations.append("Enhance repair response times for better service")
        
        return {
            'conversion_impact': conversion_impact,
            'operational_metrics': operational_metrics,
            'carbon_metrics': carbon_metrics,
            'recommendations': recommendations,
            'report_date': datetime.now().isoformat(),
            'summary': {
                'total_lights': led_count + traditional_count,
                'led_conversion_rate': operational_metrics['led_percentage'],
                'annual_carbon_savings': conversion_impact['carbon_savings_kg'],
                'annual_cost_savings': conversion_impact['cost_savings_usd'],
                'sustainability_score': min(100, (operational_metrics['led_percentage'] + 
                                                 operational_pct + repair_rate) / 3)
            }
        }
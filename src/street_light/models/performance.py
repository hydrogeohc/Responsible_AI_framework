"""
Street Light Performance Model
Neural network model for predicting street light operational status and performance metrics.
"""

import torch
import torch.nn as nn
from typing import Dict


class StreetLightPerformanceModel(nn.Module):
    """
    Neural network model for predicting street light performance metrics.
    
    Predicts operational status, energy consumption, and maintenance needs
    based on historical data and environmental factors.
    """
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 128, output_dim: int = 3):
        """
        Initialize street light performance model.
        
        Args:
            input_dim: Number of input features (e.g., age, location, usage, weather)
            hidden_dim: Hidden layer dimension
            output_dim: Number of output classes (operational/maintenance/failure)
        """
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Performance prediction head
        self.performance_head = nn.Linear(hidden_dim // 2, output_dim)
        
        # Energy consumption prediction head
        self.energy_head = nn.Linear(hidden_dim // 2, 1)
        
        # Carbon footprint prediction head
        self.carbon_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input features tensor
            
        Returns:
            Dictionary with predictions for performance, energy, and carbon
        """
        features = self.feature_extractor(x)
        
        return {
            'performance': self.performance_head(features),
            'energy_consumption': self.energy_head(features),
            'carbon_footprint': self.carbon_head(features)
        }
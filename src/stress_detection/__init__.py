"""
Stress Detection Module
A comprehensive framework for responsible AI stress detection with privacy preservation,
carbon tracking, explainability, and federated learning.
"""

from .core.framework import ResponsibleAIFramework
from .models.stress_models import StressDetectionModel, TimeSeriesStressModel, BiLSTMStressModel
from .data.data_utils import get_sample_data_for_demo

__version__ = "1.0.0"
__author__ = "Responsible AI Team"

__all__ = [
    "ResponsibleAIFramework",
    "StressDetectionModel",
    "TimeSeriesStressModel", 
    "BiLSTMStressModel",
    "get_sample_data_for_demo"
]
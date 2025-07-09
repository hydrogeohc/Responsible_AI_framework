"""
SHAP-based model explainer component.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model interpretability component.
    
    Provides feature importance analysis and explanation generation
    for machine learning model predictions.
    """
    
    def __init__(self, model: torch.nn.Module, background_data: np.ndarray):
        """
        Initialize SHAP explainer.
        
        Args:
            model: PyTorch model to explain
            background_data: Background data for SHAP explanations
        """
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
    def initialize_explainer(self, explainer_type: str = "kernel") -> bool:
        """
        Initialize SHAP explainer.
        
        Args:
            explainer_type: Type of explainer ('kernel', 'deep', 'gradient')
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if explainer_type == "kernel":
                # Model-agnostic explainer
                def model_predict(x):
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.model(torch.FloatTensor(x))
                        return torch.softmax(outputs, dim=1).numpy()
                
                self.explainer = shap.KernelExplainer(model_predict, self.background_data)
                
            elif explainer_type == "deep":
                # For neural networks
                self.explainer = shap.DeepExplainer(
                    self.model, 
                    torch.FloatTensor(self.background_data)
                )
                
            elif explainer_type == "gradient":
                # For gradient-based explanations
                self.explainer = shap.GradientExplainer(
                    self.model, 
                    torch.FloatTensor(self.background_data)
                )
                
            else:
                raise ValueError(f"Unknown explainer type: {explainer_type}")
                
            logger.info(f"Initialized SHAP {explainer_type} explainer")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            return False
    
    def explain_prediction(self, input_data: np.ndarray, 
                         feature_names: List[str] = None) -> Dict:
        """
        Generate SHAP explanations for predictions.
        
        Args:
            input_data: Input data to explain
            feature_names: Names of input features
            
        Returns:
            Dictionary with SHAP explanation results
        """
        try:
            # Initialize explainer if not already done
            if self.explainer is None:
                if not self.initialize_explainer("kernel"):
                    return {"error": "Could not initialize explainer"}
            
            # Get SHAP values
            if hasattr(self.explainer, 'shap_values'):
                shap_values = self.explainer.shap_values(input_data, nsamples=100)
            else:
                shap_values = self.explainer(torch.FloatTensor(input_data))
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # Take the first class or aggregate
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
            
            # Ensure correct shape
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Take first sample
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(shap_values))]
            
            # Create explanation dictionary
            explanation = {
                'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else list(shap_values),
                'feature_names': feature_names,
                'feature_importance': []
            }
            
            # Calculate feature importance
            for i, (name, value) in enumerate(zip(feature_names, shap_values)):
                explanation['feature_importance'].append({
                    'feature': name,
                    'shap_value': float(value),
                    'importance': abs(float(value)),
                    'impact': 'positive' if float(value) > 0 else 'negative'
                })
            
            # Sort by importance
            explanation['feature_importance'].sort(key=lambda x: x['importance'], reverse=True)
            
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {
                'error': str(e),
                'fallback_explanation': self._gradient_based_explanation(input_data, feature_names)
            }
    
    def _gradient_based_explanation(self, input_data: np.ndarray, 
                                   feature_names: List[str]) -> Dict:
        """
        Fallback gradient-based feature importance.
        
        Args:
            input_data: Input data
            feature_names: Feature names
            
        Returns:
            Dictionary with gradient-based explanation
        """
        try:
            input_tensor = torch.FloatTensor(input_data)
            input_tensor.requires_grad = True
            
            self.model.eval()
            output = self.model(input_tensor)
            
            # Get gradients
            output.backward(torch.ones_like(output))
            gradients = input_tensor.grad.abs().mean(dim=0)
            
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(gradients))]
            
            importance = []
            for i, (name, grad) in enumerate(zip(feature_names, gradients)):
                importance.append({
                    'feature': name,
                    'importance': float(grad),
                    'impact': 'gradient-based'
                })
            
            importance.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'type': 'gradient-based',
                'feature_importance': importance
            }
            
        except Exception as e:
            logger.error(f"Fallback explanation failed: {e}")
            return {'error': 'All explanation methods failed'}
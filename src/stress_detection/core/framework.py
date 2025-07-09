"""
Main Responsible AI Framework for Stress Detection
Integrates carbon tracking, privacy preservation, federated learning, and model interpretability.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from .carbon_tracker import RealCarbonTracker
from .explainer import SHAPExplainer
from .federated_learning import FlowerFederatedLearning
from .privacy_layer import PrivacyLayer
from ..models.stress_models import StressDetectionModel, TimeSeriesStressModel, BiLSTMStressModel
from ..data.data_utils import get_sample_data_for_demo

logger = logging.getLogger(__name__)


class ResponsibleAIFramework:
    """
    Main framework class that integrates all responsible AI components.
    
    Features:
    - Carbon tracking with CarbonTracker
    - Model interpretability with SHAP
    - Federated learning with Flower
    - Privacy preservation with differential privacy
    """
    
    def __init__(self, model_type: str = "simple", privacy_epsilon: float = 1.0):
        """
        Initialize the responsible AI framework.
        
        Args:
            model_type: Type of model ('simple', 'time_series', 'bilstm')
            privacy_epsilon: Privacy budget for differential privacy
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        # Initialize components
        self.carbon_tracker = RealCarbonTracker(f"StressModel_{model_type}")
        self.privacy_layer = PrivacyLayer(epsilon=privacy_epsilon)
        self.shap_explainer = None
        self.flower_fl = None
        
        # Initialize model
        self._initialize_model()
        
        # Training history
        self.training_history = []
        self.explanation_history = []
    
    def _initialize_model(self):
        """Initialize the stress detection model based on type."""
        if self.model_type == "simple":
            self.model = StressDetectionModel(input_dim=4, hidden_dim=64, num_classes=3)
        elif self.model_type == "time_series":
            self.model = TimeSeriesStressModel(input_dim=4, hidden_dim=128, num_layers=2)
        elif self.model_type == "bilstm":
            self.model = BiLSTMStressModel(input_dim=4, hidden_dim=64, num_layers=2)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_with_carbon_tracking(self, features: np.ndarray, labels: np.ndarray,
                                 epochs: int = 10, use_privacy: bool = True) -> Dict:
        """
        Train the model with carbon tracking and privacy preservation.
        
        Args:
            features: Training features
            labels: Training labels
            epochs: Number of training epochs
            use_privacy: Whether to apply differential privacy
            
        Returns:
            Training result dictionary
        """
        # Start carbon tracking
        carbon_success = self.carbon_tracker.start_tracking(max_epochs=epochs)
        
        # Apply privacy if requested
        if use_privacy:
            features = self.privacy_layer.add_differential_privacy_noise(features, sensitivity=1.0)
        
        # Prepare data
        features_scaled = self.scaler.fit_transform(features)
        
        # Training loop
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        training_losses = []
        
        for epoch in range(epochs):
            if carbon_success:
                self.carbon_tracker.epoch_start()
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(features_scaled)
            y_tensor = torch.LongTensor(labels)
            
            # Training step
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
            if carbon_success:
                self.carbon_tracker.epoch_end()
        
        # Stop carbon tracking
        carbon_result = self.carbon_tracker.stop_tracking()
        
        # Calculate final accuracy
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
        
        # Record training result
        training_record = {
            'epochs': epochs,
            'final_loss': training_losses[-1],
            'accuracy': accuracy,
            'privacy_used': use_privacy,
            'carbon_emissions': carbon_result,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(training_record)
        return training_record
    
    def predict_with_explanation(self, input_data: np.ndarray,
                               generate_explanation: bool = True) -> Dict:
        """
        Make prediction with optional SHAP explanation.
        
        Args:
            input_data: Input data for prediction
            generate_explanation: Whether to generate SHAP explanation
            
        Returns:
            Prediction result with optional explanation
        """
        # Prepare input
        input_normalized = self.scaler.transform(input_data)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_normalized)
            prediction = self.model(input_tensor)
            stress_probs = torch.softmax(prediction, dim=1)
            predicted_class = torch.argmax(stress_probs, dim=1)
        
        # Prepare result
        stress_levels = ["Low Stress", "Medium Stress", "High Stress"]
        result = {
            'predicted_stress': stress_levels[predicted_class.item()],
            'confidence': stress_probs[0][predicted_class].item(),
            'probabilities': {
                'low': stress_probs[0][0].item(),
                'medium': stress_probs[0][1].item(),
                'high': stress_probs[0][2].item()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add explanation if requested
        if generate_explanation:
            explanation = self._generate_explanation(input_data)
            result['explanation'] = explanation
        
        return result
    
    def _generate_explanation(self, input_data: np.ndarray) -> Dict:
        """Generate SHAP explanation for prediction."""
        try:
            # Initialize SHAP explainer if not already done
            if self.shap_explainer is None:
                background_data, _ = get_sample_data_for_demo()
                background_data = self.scaler.transform(background_data[:50])
                self.shap_explainer = SHAPExplainer(self.model, background_data)
            
            # Generate explanation
            feature_names = ['Age', 'Height (cm)', 'Weight (kg)', 'Physical Activity']
            explanation = self.shap_explainer.explain_prediction(input_data, feature_names)
            
            # Store explanation
            self.explanation_history.append({
                'input_data': input_data.tolist(),
                'explanation': explanation,
                'timestamp': datetime.now().isoformat()
            })
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {'error': str(e)}
    
    def federated_learning(self, client_data: List[Tuple[np.ndarray, np.ndarray]],
                          rounds: int = 3) -> Dict:
        """
        Perform federated learning with privacy preservation.
        
        Args:
            client_data: List of (features, labels) tuples for each client
            rounds: Number of federated learning rounds
            
        Returns:
            Federated learning result
        """
        # Initialize Flower federated learning
        model_params = {
            'input_dim': 4,
            'hidden_dim': 64,
            'num_classes': 3
        }
        
        self.flower_fl = FlowerFederatedLearning(self.model.__class__, model_params)
        
        # Apply privacy to client data
        privacy_protected_data = []
        for features, labels in client_data:
            if len(features) > 0:
                private_features = self.privacy_layer.add_differential_privacy_noise(
                    features, sensitivity=0.5
                )
                privacy_protected_data.append((private_features, labels))
            else:
                privacy_protected_data.append((features, labels))
        
        # Start carbon tracking
        carbon_success = self.carbon_tracker.start_tracking(max_epochs=rounds)
        
        # Run federated training
        federated_result = self.flower_fl.simulate_federated_training(
            privacy_protected_data, rounds
        )
        
        # Stop carbon tracking
        carbon_result = self.carbon_tracker.stop_tracking()
        
        # Update local model with federated result
        if federated_result.get('success', False):
            self.model.load_state_dict(federated_result['final_model'])
        
        # Return complete result
        return {
            'federated_training': federated_result,
            'carbon_emissions': carbon_result,
            'privacy_protected': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict:
        """Get comprehensive framework status."""
        return {
            'model_type': self.model_type,
            'model_trained': len(self.training_history) > 0,
            'training_history': len(self.training_history),
            'explanations_generated': len(self.explanation_history),
            'carbon_tracking_active': self.carbon_tracker.tracking_active,
            'privacy_epsilon': self.privacy_layer.epsilon,
            'shap_explainer_initialized': self.shap_explainer is not None,
            'flower_fl_initialized': self.flower_fl is not None,
            'last_updated': datetime.now().isoformat()
        }
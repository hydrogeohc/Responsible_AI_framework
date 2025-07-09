"""
Neural network models for stress detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StressDetectionModel(nn.Module):
    """
    Simple feedforward neural network for stress detection.
    
    Args:
        input_dim (int): Number of input features (default: 4)
        hidden_dim (int): Hidden layer dimension (default: 64)
        num_classes (int): Number of output classes (default: 3)
    """
    
    def __init__(self, input_dim=4, hidden_dim=64, num_classes=3):
        super(StressDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class TimeSeriesStressModel(nn.Module):
    """
    LSTM-based model with attention mechanism for time series stress detection.
    
    Args:
        input_dim (int): Number of input features per timestep
        hidden_dim (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes
        sequence_length (int): Length of input sequences
    """
    
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2, num_classes=3, sequence_length=60):
        super(TimeSeriesStressModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # LSTM for time series processing
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            tuple: (output, attention_weights)
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classification
        output = self.classifier(attended_features)
        
        return output, attention_weights.squeeze(-1)


class BiLSTMStressModel(nn.Module):
    """
    Bidirectional LSTM model for stress detection.
    
    Args:
        input_dim (int): Number of input features per timestep
        hidden_dim (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes
    """
    
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, num_classes=3):
        super(BiLSTMStressModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=0.2, bidirectional=True)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Classification output
        """
        batch_size = x.size(0)
        
        # BiLSTM forward pass
        lstm_out, (h_n, c_n) = self.bilstm(x)
        
        # Use last hidden state from both directions
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Feature extraction
        features = self.feature_extractor(last_hidden)
        
        # Classification
        output = self.classifier(features)
        
        return output
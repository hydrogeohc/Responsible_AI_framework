# 🧠 Responsible AI Framework

A comprehensive framework for responsible AI with stress detection models, integrating carbon tracking, privacy preservation, federated learning, and model interpretability.

## ✨ Features

- **🌱 Carbon Tracking**: Real-time CO₂ emissions monitoring with CarbonTracker
- **🔍 Model Interpretability**: SHAP-based explanations for all predictions
- **🌸 Federated Learning**: Privacy-preserving distributed training with Flower
- **🔒 Privacy Protection**: Differential privacy and secure computation
- **🧠 Multiple Models**: Simple NN, LSTM with attention, and BiLSTM architectures

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Framework**:
   ```bash
   python quick_start.py
   ```

3. **Launch Web Interface**:
   ```bash
   streamlit run main_app.py
   ```

## 📁 Project Structure

```
├── main_app.py              # Main Streamlit application
├── quick_start.py           # Quick start demo script
├── requirements.txt         # Python dependencies
├── src/                     # Source code
│   ├── core/               # Core framework components
│   │   ├── framework.py    # Main framework class
│   │   ├── carbon_tracker.py  # Carbon tracking
│   │   ├── explainer.py    # SHAP explanations
│   │   ├── federated_learning.py  # Flower FL
│   │   └── privacy_layer.py    # Privacy protection
│   ├── models/             # Neural network models
│   │   └── stress_models.py    # Stress detection models
│   └── data/               # Data utilities
│       ├── data_utils.py   # Data loading and processing
│       └── time_series_utils.py  # Time series utilities
├── wearable_dataset/       # Wearable sensor dataset
└── saved_models/          # Trained model checkpoints
```

## 🎯 Usage Examples

### Basic Training and Prediction
```python
from src.core.framework import ResponsibleAIFramework

# Initialize framework
framework = ResponsibleAIFramework(model_type="simple", privacy_epsilon=1.0)

# Train with carbon tracking
training_result = framework.train_with_carbon_tracking(features, labels)

# Make prediction with explanation
prediction = framework.predict_with_explanation(input_data)
```

### Federated Learning
```python
# Prepare client data
client_data = [(features1, labels1), (features2, labels2), (features3, labels3)]

# Run federated learning
fed_result = framework.federated_learning(client_data, rounds=5)
```

## 📊 Package Integration

- **CarbonTracker**: Real carbon emissions monitoring
- **SHAP**: Model interpretability and explanations
- **Flower**: Federated learning orchestration
- **Privacy Layer**: Differential privacy implementation

## 🛠️ Development

The framework is designed to be modular and extensible. Each component can be used independently or as part of the complete system.

## 📝 License

This project is for educational and research purposes.

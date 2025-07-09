# 🏙️ Responsible AI Framework for Smart City Street Lighting

A comprehensive framework that integrates responsible AI principles including carbon tracking, explainability, federated learning, and privacy preservation for both stress detection and smart city street lighting systems.

## 🚀 Key Features

### 🤖 **Multi-Domain AI Models**
- **Stress Detection**: Simple NN, LSTM with attention, and BiLSTM architectures for wearable sensor data
- **Street Light Performance**: Predicts operational status, energy consumption, and maintenance needs
- **Carbon Footprint**: Specialized carbon emissions tracking and optimization
- **LED Conversion**: Time series optimization for LED conversion scheduling

### 🌱 **Carbon Tracking**
- Real-time CO₂ emissions monitoring during model training with CarbonTracker
- Carbon savings calculations from LED conversions
- Sustainability metrics and comprehensive reporting
- Energy efficiency analysis and optimization

### 🔍 **Explainable AI**
- SHAP-based explanations for all predictions
- Feature importance analysis for maintenance decisions
- Transparent AI decision-making process
- Multiple explanation methods (kernel, gradient, deep)

### 🔒 **Privacy Preservation**
- Differential privacy with configurable epsilon
- K-anonymity for location and sensitive data
- Homomorphic encryption simulation
- Privacy budget tracking and management

### 🌐 **Federated Learning**
- Flower-based federated learning across city districts
- Privacy-preserving model updates
- Secure aggregation of distributed data
- Multi-client collaboration without centralized data

## 📁 Project Structure

```
├── main_app.py                    # Main Streamlit application
├── quick_start.py                 # Quick start demo script
├── street_light_demo.py           # Street light comprehensive demo
├── test_integration.py            # Integration tests
├── requirements.txt               # Python dependencies
├── src/                          # Source code
│   ├── stress_detection/         # Stress detection system
│   │   ├── __init__.py           # Module exports
│   │   ├── core/                 # Core responsible AI components
│   │   │   ├── framework.py      # Main ResponsibleAIFramework class
│   │   │   ├── carbon_tracker.py # Carbon tracking with CarbonTracker
│   │   │   ├── explainer.py      # SHAP explanations
│   │   │   ├── federated_learning.py # Flower federated learning
│   │   │   └── privacy_layer.py  # Privacy protection layer
│   │   ├── models/               # Neural network models
│   │   │   └── stress_models.py  # Stress detection models
│   │   └── data/                 # Data utilities
│   │       ├── data_utils.py     # Data loading and processing
│   │       └── time_series_utils.py # Time series utilities
│   └── street_light/             # Street light IoT system
│       ├── __init__.py           # Module exports
│       ├── core/                 # Core framework components
│       │   └── framework.py      # StreetLightResponsibleAI class
│       ├── models/               # Neural network models
│       │   ├── performance.py    # Performance prediction model
│       │   ├── carbon.py         # Carbon footprint model
│       │   └── led_conversion.py # LED conversion optimization
│       ├── data/                 # Data processing modules
│       │   ├── processor.py      # Data preprocessing
│       │   └── loader.py         # Data loading utilities
│       └── utils/                # Utility functions
│           ├── helpers.py        # Helper functions
│           ├── constants.py      # Constants and configuration
│           └── metrics.py        # Performance metrics
├── config/                       # Configuration files
├── smart_city_light/             # Street light dataset
├── wearable_dataset/             # Wearable sensor dataset
├── saved_models/                 # Trained model checkpoints
└── carbon_logs/                  # Carbon tracking logs
```

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Clone the repository
git clone <repository-url>
cd Responsible_AI_framework

# Install dependencies
pip install -r requirements.txt
```

### Running the Framework
All demos and applications should be run from the project root directory:
```bash
# Quick start demo
python quick_start.py

# Comprehensive stress detection demo
python stress_detection_demo.py

# Street light IoT demo
python street_light_demo.py

# Web interface
streamlit run main_app.py

# Integration tests
python test_integration.py
```

### Key Dependencies
- `torch>=2.0.0` - Deep learning framework
- `shap>=0.48.0` - Model explainability
- `flwr>=1.19.0` - Federated learning
- `carbontracker>=2.3.0` - Carbon tracking
- `syft>=0.8.0` - Privacy-preserving ML
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - ML utilities
- `streamlit>=1.20.0` - Web interface
- `plotly>=5.0.0` - Interactive visualizations
- `pytest>=7.0.0` - Testing framework
- `jupyterlab>=3.0.0` - Development environment

### Configuration
The framework supports multiple environments:
- **Development**: Fast training, relaxed privacy
- **Production**: Thorough training, strict privacy
- **Testing**: Minimal training for CI/CD

```python
from config import get_config

# Get configuration for specific environment
config = get_config('dev')  # or 'prod', 'test'
```

## 🎯 Usage Examples

### **Stress Detection (Original Framework)**
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.stress_detection.core.framework import ResponsibleAIFramework

# Initialize framework
framework = ResponsibleAIFramework(model_type="simple", privacy_epsilon=1.0)

# Train with carbon tracking
training_result = framework.train_with_carbon_tracking(features, labels)

# Make prediction with explanation
prediction = framework.predict_with_explanation(input_data)

# Federated learning
client_data = [(features1, labels1), (features2, labels2)]
fed_result = framework.federated_learning(client_data, rounds=5)
```

### **Street Light System**
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.street_light import StreetLightResponsibleAI, StreetLightDataProcessor, create_sample_district_data
from config import get_config

# Initialize framework
config = get_config('dev')
framework = StreetLightResponsibleAI(config=config)

# Load data
data_processor = StreetLightDataProcessor()
street_light_data = data_processor.load_street_light_data('data.csv')

# Train model with carbon tracking
result = framework.train_performance_model(street_light_data, epochs=10)

# Make predictions with explanations
prediction = framework.predict_with_explanation(
    street_light_data, 
    model_type="performance"
)

# Federated learning across districts
district_data = create_sample_district_data(n_districts=3)
federated_result = framework.federated_learning_simulation(
    district_data, rounds=5, model_type="performance"
)

# Privacy analysis
privacy_report = framework.privacy_analysis(street_light_data)

# Carbon footprint report
carbon_report = framework.get_carbon_report()
```

## 📱 Demo Applications

### **Quick Start Demo**
```bash
python quick_start.py
```

### **Web Interface**
```bash
streamlit run main_app.py
```

### **Street Light Demo**
```bash
python street_light_demo.py
```

## 📊 Data Integration

### **Supported Data Sources**
- **Wearable Sensors**: Stress detection from physiological data
- **LED Conversion Data**: Historical LED light installation records
- **Outage Repair Data**: Maintenance response time metrics
- **Performance Metrics**: Operational efficiency indicators

### **Data Processing Pipeline**
1. **Data Loading**: Automated loading from CSV files with validation
2. **Feature Engineering**: Extraction of relevant features for ML models
3. **Privacy Protection**: Application of differential privacy and k-anonymity
4. **Scaling**: Normalization and standardization for model training

## 📈 Model Architectures

### **Stress Detection Models**
```python
# Simple Neural Network
Input: [Age, Height, Weight, Physical Activity]
Output: [Low Stress, Medium Stress, High Stress]

# LSTM with Attention
Input: Time series of physiological data
Output: Stress level predictions with temporal patterns

# BiLSTM
Input: Bidirectional sequence processing
Output: Enhanced stress detection with context
```

### **Street Light Models**
```python
# Performance Model
Input: [LED Count, Operational %, Repair Rate, Energy Usage, 
        Age, Location Density, Weather Factor, Usage Pattern]
Output: {
    'performance': [Operational, Maintenance, Failure],
    'energy_consumption': float,
    'carbon_footprint': float
}

# Carbon Model
Input: [Performance Features + Base Energy, Carbon Intensity, 
        Grid Efficiency, Renewable %]
Output: Carbon emissions (kg CO2)

# LED Conversion Model
Input: Time series of conversion metrics
Output: [Optimal conversion rate, Energy savings]
```

## 🔧 Configuration Options

### **Model Parameters**
```python
'model_params': {
    'performance': {
        'INPUT_DIM': 8,
        'HIDDEN_DIM': 128,
        'OUTPUT_DIM': 3,
        'DROPOUT_RATE': 0.2
    },
    'carbon': {
        'INPUT_DIM': 10,
        'HIDDEN_DIM': 96
    }
}
```

### **Privacy Settings**
```python
'privacy_epsilon': 1.0,        # Differential privacy budget
'k_anonymity': 5,              # K-anonymity parameter
```

### **Training Configuration**
```python
'training': {
    'LEARNING_RATE': 0.001,
    'BATCH_SIZE': 32,
    'MAX_EPOCHS': 100,
    'VALIDATION_SPLIT': 0.2
}
```

## 📊 Metrics & Reporting

### **Performance Metrics**
- **Operational Efficiency**: Percentage of lights functioning
- **Maintenance Response**: Repair completion rates
- **Energy Consumption**: kWh usage tracking
- **Carbon Footprint**: CO2 emissions monitoring
- **Model Accuracy**: Prediction performance metrics

### **Sustainability Metrics**
- **Energy Savings**: LED vs traditional light consumption
- **Carbon Reduction**: CO2 emissions prevented
- **Cost Savings**: Financial impact of LED conversion
- **Sustainability Score**: Overall environmental impact

## 🔍 Explainability Features

### **SHAP Explanations**
- **Feature Importance**: Which factors most influence predictions
- **Local Explanations**: Why specific predictions were made
- **Global Patterns**: Overall model behavior insights

### **Explanation Types**
- **Kernel Explainer**: Model-agnostic explanations
- **Deep Explainer**: Neural network-specific insights
- **Gradient Explainer**: Gradient-based feature importance

## 🌐 Federated Learning

### **Multi-Client Training**
- **Privacy-Preserving**: No raw data sharing between clients
- **Collaborative Learning**: Shared model improvements
- **Secure Aggregation**: FedAvg algorithm implementation

### **Client Configuration**
```python
'federated_learning': {
    'DEFAULT_ROUNDS': 3,
    'LOCAL_EPOCHS': 5,
    'MIN_CLIENTS': 2,
    'MAX_CLIENTS': 10
}
```

## 🔒 Privacy & Security

### **Differential Privacy**
- **Noise Addition**: Laplace mechanism for privacy
- **Budget Tracking**: Epsilon consumption monitoring
- **Composition**: Privacy budget management across queries

### **Additional Privacy Features**
- **K-Anonymity**: Location data generalization
- **Homomorphic Encryption**: Computation on encrypted data
- **Secure Aggregation**: Privacy-preserving federated averaging

## 🎯 Use Cases

### **Healthcare Applications**
- **Stress Monitoring**: Real-time stress detection from wearable devices
- **Privacy-Safe Analytics**: Analyze health patterns without compromising privacy
- **Federated Health Research**: Collaborative research without data sharing

### **Smart City Applications**
- **Predictive Maintenance**: Predict when street lights need maintenance
- **Energy Optimization**: Optimize LED conversion schedules
- **Carbon Tracking**: Monitor environmental impact of lighting systems
- **Privacy-Safe Analytics**: Analyze patterns without compromising privacy

### **Research Applications**
- **Responsible AI**: Study ethical AI implementation across domains
- **Federated Learning**: Research collaborative learning without data sharing
- **Explainable AI**: Understand AI decision-making in critical applications
- **Carbon-Aware Computing**: Measure and reduce AI environmental impact

## 🛠️ Troubleshooting

### **Common Import Issues**
If you encounter import errors, ensure:

1. **Run from project root**: All scripts must be run from the main project directory
2. **Python path**: The `src` directory is automatically added to Python path by each script
3. **Dependencies**: Install all requirements with `pip install -r requirements.txt`

### **Carbon Tracker Issues**
- On macOS, carbon tracking may require sudo permissions for power metrics
- This is normal and doesn't affect core functionality
- The framework will continue to work with estimated carbon values

### **Federated Learning Setup**
- Flower federated learning runs in simulation mode by default
- For production deployment, configure actual client endpoints
- Privacy settings can be adjusted in the configuration files

## 🤝 Contributing

### **Development Setup**
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python test_integration.py`
4. Follow the organized module structure

### **Code Organization**
- **Core Components**: Add to `src/core/`
- **Street Light Models**: Add to `src/street_light/models/`
- **Data Processing**: Extend `src/street_light/data/`
- **Utilities**: Add helpers to `src/street_light/utils/`
- **Configuration**: Update `config/street_light_config.py`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CarbonTracker**: For carbon emissions tracking
- **SHAP**: For model explainability
- **Flower**: For federated learning framework
- **PySyft**: For privacy-preserving machine learning
- **Street Light Data**: City of Los Angeles Open Data Portal
- **Wearable Dataset**: For stress detection research

## 📧 Contact

For questions and support, please open an issue in the repository.

---

*This framework demonstrates the integration of responsible AI principles across multiple domains, showcasing how carbon tracking, explainability, federated learning, and privacy preservation can be implemented in both healthcare and smart city applications.*
#!/usr/bin/env python3
"""
Cleanup script to organize the codebase and remove redundant files.
"""

import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_codebase():
    """Clean up the codebase by removing redundant files and organizing structure."""
    
    # Files to keep (clean, organized versions)
    keep_files = {
        'main_app.py',
        'src/',
        'wearable_dataset/',
        'INTEGRATION_SUMMARY.md',
        'README.markdown',
        'requirements.txt',
        'saved_models/',
        'myenv/',
        'cleanup.py'
    }
    
    # Files to remove (redundant or old versions)
    remove_files = [
        'app.py',  # Replaced by main_app.py
        'enhanced_app.py',  # Replaced by main_app.py
        'integrated_app.py',  # Replaced by main_app.py
        'models.py',  # Moved to src/models/
        'data_utils.py',  # Moved to src/data/
        'time_series_utils.py',  # Moved to src/data/
        'enhanced_framework.py',  # Replaced by src/core/
        'integrated_stress_framework.py',  # Replaced by src/core/
        'carbon_emission_layer.py',  # Replaced by src/core/carbon_tracker.py
        'privacy_utils.py',  # Replaced by src/core/privacy_layer.py
        'security_layer.py',  # Functionality moved to src/core/
        'interpretability_utils.py',  # Replaced by src/core/explainer.py
        'client.py',  # Old federated learning code
        'server.py',  # Old federated learning code
        'demo_integration.py',  # Old demo
        'demo_enhanced_framework.py',  # Old demo
        'test_enhanced_framework.py',  # Old test
        'simple_demo.py',  # Old demo
        'train_time_series.py',  # Old training script
        'deploy.py',  # Old deployment script
        'README_DEPLOYMENT.md',  # Old deployment docs
        'deployment.log',  # Old log file
        'enhanced_framework_test_report.json'  # Old test report
    ]
    
    # Directories to remove
    remove_dirs = [
        'carbon_logs/',
        'test_carbon/',
        '__pycache__/'
    ]
    
    logger.info("Starting codebase cleanup...")
    
    # Remove redundant files
    for file_name in remove_files:
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                logger.info(f"Removed file: {file_name}")
            except Exception as e:
                logger.warning(f"Could not remove {file_name}: {e}")
    
    # Remove redundant directories
    for dir_name in remove_dirs:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                logger.info(f"Removed directory: {dir_name}")
            except Exception as e:
                logger.warning(f"Could not remove {dir_name}: {e}")
    
    # Create clean requirements.txt
    create_clean_requirements()
    
    # Create clean README
    create_clean_readme()
    
    # Create quick start script
    create_quick_start()
    
    logger.info("Codebase cleanup completed!")
    logger.info("✅ Clean structure created with organized modules")
    logger.info("🚀 Run 'python quick_start.py' to test the framework")
    logger.info("🌐 Run 'streamlit run main_app.py' for the web interface")

def create_clean_requirements():
    """Create a clean requirements.txt file."""
    requirements = """# Core ML and data science
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Streamlit for web app
streamlit>=1.20.0
plotly>=5.0.0

# Responsible AI packages
carbontracker>=2.3.0
shap>=0.48.0
flwr>=1.19.0

# Additional utilities
pyyaml>=6.0.0
tqdm>=4.60.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    logger.info("Created clean requirements.txt")

def create_clean_readme():
    """Create a clean README file."""
    readme_content = """# 🧠 Responsible AI Framework

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
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    logger.info("Created clean README.md")

def create_quick_start():
    """Create a quick start demo script."""
    quick_start_content = """#!/usr/bin/env python3
\"\"\"
Quick start demo for the Responsible AI Framework.
\"\"\"

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.framework import ResponsibleAIFramework
from data.data_utils import get_sample_data_for_demo

def main():
    print("🧠 Responsible AI Framework - Quick Start Demo")
    print("=" * 60)
    
    # Initialize framework
    print("\\n1. Initializing framework...")
    framework = ResponsibleAIFramework(model_type="simple", privacy_epsilon=1.0)
    print("✅ Framework initialized")
    
    # Load sample data
    print("\\n2. Loading sample data...")
    features, labels = get_sample_data_for_demo()
    print(f"✅ Loaded {len(features)} samples")
    
    # Train model
    print("\\n3. Training model with carbon tracking...")
    training_result = framework.train_with_carbon_tracking(
        features, labels, epochs=5, use_privacy=True
    )
    print(f"✅ Training completed - Accuracy: {training_result['accuracy']:.1%}")
    
    # Make prediction
    print("\\n4. Making prediction with explanation...")
    test_input = np.array([[30, 170, 70, 1]])  # Age, Height, Weight, Activity
    prediction = framework.predict_with_explanation(test_input)
    
    print(f"✅ Prediction: {prediction['predicted_stress']}")
    print(f"   Confidence: {prediction['confidence']:.1%}")
    
    if 'explanation' in prediction and 'feature_importance' in prediction['explanation']:
        print("   Feature importance:")
        for feat in prediction['explanation']['feature_importance'][:3]:
            print(f"     • {feat['feature']}: {feat['shap_value']:.3f}")
    
    # Show framework status
    print("\\n5. Framework status:")
    status = framework.get_status()
    print(f"   Model trained: {status['model_trained']}")
    print(f"   Carbon tracking: {status['carbon_tracking_active']}")
    print(f"   Privacy epsilon: {status['privacy_epsilon']}")
    
    print("\\n🎉 Quick start demo completed!")
    print("\\nNext steps:")
    print("- Run 'streamlit run main_app.py' for the web interface")
    print("- Explore the src/ directory for framework components")
    print("- Check INTEGRATION_SUMMARY.md for detailed documentation")

if __name__ == "__main__":
    main()
"""
    
    with open('quick_start.py', 'w') as f:
        f.write(quick_start_content)
    
    logger.info("Created quick_start.py")

if __name__ == "__main__":
    cleanup_codebase()
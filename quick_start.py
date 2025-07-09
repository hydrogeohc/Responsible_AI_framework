#!/usr/bin/env python3
"""
Quick start demo for the Responsible AI Framework.
"""

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
    print("\n1. Initializing framework...")
    framework = ResponsibleAIFramework(model_type="simple", privacy_epsilon=1.0)
    print("✅ Framework initialized")
    
    # Load sample data
    print("\n2. Loading sample data...")
    features, labels = get_sample_data_for_demo()
    print(f"✅ Loaded {len(features)} samples")
    
    # Train model
    print("\n3. Training model with carbon tracking...")
    training_result = framework.train_with_carbon_tracking(
        features, labels, epochs=5, use_privacy=True
    )
    print(f"✅ Training completed - Accuracy: {training_result['accuracy']:.1%}")
    
    # Make prediction
    print("\n4. Making prediction with explanation...")
    test_input = np.array([[30, 170, 70, 1]])  # Age, Height, Weight, Activity
    prediction = framework.predict_with_explanation(test_input)
    
    print(f"✅ Prediction: {prediction['predicted_stress']}")
    print(f"   Confidence: {prediction['confidence']:.1%}")
    
    if 'explanation' in prediction and 'feature_importance' in prediction['explanation']:
        print("   Feature importance:")
        for feat in prediction['explanation']['feature_importance'][:3]:
            print(f"     • {feat['feature']}: {feat['shap_value']:.3f}")
    
    # Show framework status
    print("\n5. Framework status:")
    status = framework.get_status()
    print(f"   Model trained: {status['model_trained']}")
    print(f"   Carbon tracking: {status['carbon_tracking_active']}")
    print(f"   Privacy epsilon: {status['privacy_epsilon']}")
    
    print("\n🎉 Quick start demo completed!")
    print("\nNext steps:")
    print("- Run 'streamlit run main_app.py' for the web interface")
    print("- Explore the src/ directory for framework components")
    print("- Check INTEGRATION_SUMMARY.md for detailed documentation")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick start demo for the Responsible AI Framework - Stress Detection System.
Demonstrates the core functionality of the stress detection framework with
carbon tracking, privacy preservation, and model explainability.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.stress_detection.core.framework import ResponsibleAIFramework
    from src.stress_detection.data.data_utils import get_sample_data_for_demo
    print("✅ Successfully imported stress detection components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

def main():
    """Main demonstration function for stress detection framework."""
    print("🧠 Responsible AI Framework - Stress Detection Quick Start")
    print("=" * 65)
    
    # Initialize framework
    print("\n1. 🚀 Initializing Responsible AI Framework...")
    try:
        framework = ResponsibleAIFramework(model_type="simple", privacy_epsilon=1.0)
        print("   ✅ Framework initialized successfully")
        print(f"   📊 Model type: Simple Neural Network")
        print(f"   🔒 Privacy epsilon: 1.0 (differential privacy enabled)")
    except Exception as e:
        print(f"   ❌ Framework initialization failed: {e}")
        return
    
    # Load sample data
    print("\n2. 📊 Loading Sample Stress Detection Data...")
    try:
        features, labels = get_sample_data_for_demo()
        print(f"   ✅ Loaded {len(features)} samples with {features.shape[1]} features")
        print(f"   📝 Features: Age, Height (cm), Weight (kg), Physical Activity")
        print(f"   🎯 Labels: {len(np.unique(labels))} stress levels (Low, Medium, High)")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return
    
    # Train model with carbon tracking
    print("\n3. 🌱 Training Model with Carbon Tracking...")
    try:
        training_result = framework.train_with_carbon_tracking(
            features, labels, epochs=5, use_privacy=True
        )
        
        if 'error' in training_result:
            print(f"   ❌ Training failed: {training_result['error']}")
            return
        
        print(f"   ✅ Training completed successfully!")
        print(f"   📈 Final accuracy: {training_result['accuracy']:.1%}")
        print(f"   🌱 Carbon emissions: {training_result['carbon_emissions']['co2_kg']:.6f} kg CO2")
        print(f"   ⚡ Energy consumption: {training_result['carbon_emissions']['energy_kwh']:.6f} kWh")
        print(f"   🔒 Privacy protection: {'Applied' if training_result['privacy_used'] else 'Not used'}")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return
    
    # Make prediction with explanation
    print("\n4. 🔍 Making Prediction with SHAP Explanation...")
    try:
        # Example input: 30-year-old, 170cm, 70kg, moderate activity
        test_input = np.array([[30, 170, 70, 1]])
        
        print(f"   📝 Input data: Age=30, Height=170cm, Weight=70kg, Activity=1")
        
        prediction = framework.predict_with_explanation(test_input)
        
        if 'error' in prediction:
            print(f"   ❌ Prediction failed: {prediction['error']}")
            return
        
        print(f"   ✅ Prediction completed!")
        print(f"   🎯 Predicted stress level: {prediction['predicted_stress']}")
        print(f"   📊 Confidence: {prediction['confidence']:.1%}")
        
        # Show probability distribution
        if 'probabilities' in prediction:
            probs = prediction['probabilities']
            print(f"   📈 Probability distribution:")
            print(f"      • Low stress: {probs['low']:.1%}")
            print(f"      • Medium stress: {probs['medium']:.1%}")
            print(f"      • High stress: {probs['high']:.1%}")
        
        # Show SHAP explanation
        if 'explanation' in prediction and 'feature_importance' in prediction['explanation']:
            print(f"   🔍 Feature importance (SHAP values):")
            for i, feat in enumerate(prediction['explanation']['feature_importance'][:4]):
                impact_icon = "📈" if feat['impact'] == 'positive' else "📉"
                print(f"      {i+1}. {feat['feature']}: {feat['shap_value']:.3f} {impact_icon}")
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return
    
    # Test federated learning simulation
    print("\n5. 🌐 Testing Federated Learning Simulation...")
    try:
        # Create sample client data
        client_data = []
        for i in range(3):
            # Generate different client data
            client_features = features[i*20:(i+1)*20] + np.random.normal(0, 0.1, (20, 4))
            client_labels = labels[i*20:(i+1)*20]
            client_data.append((client_features, client_labels))
        
        print(f"   📊 Simulating {len(client_data)} federated clients...")
        
        fed_result = framework.federated_learning(client_data, rounds=2)
        
        if 'error' in fed_result:
            print(f"   ❌ Federated learning failed: {fed_result['error']}")
        else:
            print(f"   ✅ Federated learning completed!")
            print(f"   🤝 Clients: {len(client_data)}")
            print(f"   🔄 Rounds: 2")
            print(f"   🔒 Privacy protected: {fed_result['privacy_protected']}")
            print(f"   🌱 Carbon emissions: {fed_result['carbon_emissions']['co2_kg']:.6f} kg CO2")
        
    except Exception as e:
        print(f"   ⚠️  Federated learning simulation failed: {e}")
        print(f"   ℹ️  This is optional - core functionality still works")
    
    # Show framework status
    print("\n6. 📋 Framework Status Summary...")
    try:
        status = framework.get_status()
        print(f"   Model trained: {'✅ Yes' if status['model_trained'] else '❌ No'}")
        print(f"   Training sessions: {status['training_history']}")
        print(f"   Explanations generated: {status['explanations_generated']}")
        print(f"   Carbon tracking: {'🌱 Active' if status['carbon_tracking_active'] else '❌ Inactive'}")
        print(f"   Privacy epsilon: 🔒 {status['privacy_epsilon']}")
        print(f"   SHAP explainer: {'🔍 Ready' if status['shap_explainer_initialized'] else '⚠️ Not initialized'}")
        
    except Exception as e:
        print(f"   ⚠️  Could not get status: {e}")
    
    # Completion message
    print("\n🎉 Stress Detection Quick Start Demo Completed!")
    print("=" * 65)
    print("\n🚀 Next Steps:")
    print("   • Run 'streamlit run main_app.py' for the web interface")
    print("   • Try 'python street_light_demo.py' for the IoT street light demo")
    print("   • Run 'python test_integration.py' to test all components")
    print("   • Explore the src/stress_detection/ directory for framework components")
    
    print("\n📚 Key Features Demonstrated:")
    print("   ✅ Neural network training with carbon tracking")
    print("   ✅ Differential privacy for data protection")
    print("   ✅ SHAP-based model explainability")
    print("   ✅ Federated learning simulation")
    print("   ✅ Comprehensive responsible AI metrics")
    
    print(f"\n⏰ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
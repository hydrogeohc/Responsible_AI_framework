"""
Stress Detection Responsible AI Framework Demo
Comprehensive demonstration of all stress detection features including:
- Multiple model architectures (Simple NN, LSTM, BiLSTM)
- Carbon tracking during training
- SHAP explainability for predictions
- Federated learning across healthcare institutions
- Privacy preservation with differential privacy
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import organized modules
try:
    from src.stress_detection.core.framework import ResponsibleAIFramework
    from src.stress_detection.data.data_utils import get_sample_data_for_demo
    from src.stress_detection.data.time_series_utils import get_sample_time_series_for_demo
    print("✅ Successfully imported stress detection components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


def create_sample_healthcare_data(n_institutions: int = 3, samples_per_institution: int = 50):
    """
    Create sample data for multiple healthcare institutions.
    
    Args:
        n_institutions: Number of healthcare institutions
        samples_per_institution: Number of samples per institution
        
    Returns:
        List of DataFrames for each institution
    """
    institution_data = []
    
    for i in range(n_institutions):
        # Create institution-specific data with some variation
        np.random.seed(42 + i)  # Different seed for each institution
        
        # Generate features with institutional bias
        ages = np.random.normal(35 + i * 5, 10, samples_per_institution)
        heights = np.random.normal(170 + i * 2, 8, samples_per_institution)
        weights = np.random.normal(70 + i * 3, 12, samples_per_institution)
        activity = np.random.randint(0, 3, samples_per_institution)
        
        # Generate labels with different stress patterns per institution
        stress_prob = 0.3 + i * 0.1  # Different stress rates
        labels = np.random.choice([0, 1, 2], samples_per_institution, 
                                p=[1-stress_prob, stress_prob*0.6, stress_prob*0.4])
        
        features = np.column_stack([ages, heights, weights, activity])
        institution_data.append((features, labels))
    
    return institution_data


def main():
    """Main comprehensive demonstration function."""
    print("🧠 Stress Detection Responsible AI Framework - Comprehensive Demo")
    print("=" * 70)
    
    # Initialize the framework
    print("\n1. 🚀 Initializing Responsible AI Framework...")
    try:
        framework = ResponsibleAIFramework(model_type="simple", privacy_epsilon=1.0)
        print("   ✅ Framework initialized successfully")
        
        # Display framework status
        status = framework.get_status()
        print(f"   📊 Model type: {status.get('model_type', 'simple')}")
        print(f"   🔒 Privacy epsilon: {status.get('privacy_epsilon', 1.0)}")
        print(f"   🌱 Carbon tracking: {'Active' if status.get('carbon_tracking_active', False) else 'Ready'}")
        print(f"   🔍 Explainability: {'Ready' if status.get('shap_explainer_initialized', False) else 'Will initialize'}")
        
    except Exception as e:
        print(f"   ❌ Framework initialization failed: {e}")
        return
    
    # Load stress detection data
    print("\n2. 📊 Loading Stress Detection Data...")
    try:
        features, labels = get_sample_data_for_demo()
        print(f"   ✅ Loaded {len(features)} samples with {features.shape[1]} features")
        print(f"   📝 Features: Age, Height (cm), Weight (kg), Physical Activity Level")
        print(f"   🎯 Stress levels: {len(np.unique(labels))} classes")
        
        # Show data statistics
        print(f"   📈 Data statistics:")
        print(f"      • Age range: {features[:, 0].min():.1f} - {features[:, 0].max():.1f} years")
        print(f"      • Height range: {features[:, 1].min():.1f} - {features[:, 1].max():.1f} cm")
        print(f"      • Weight range: {features[:, 2].min():.1f} - {features[:, 2].max():.1f} kg")
        print(f"      • Activity levels: {int(features[:, 3].min())} - {int(features[:, 3].max())}")
        
        # Show stress distribution
        unique, counts = np.unique(labels, return_counts=True)
        stress_labels = ["Low Stress", "Medium Stress", "High Stress"]
        print(f"   📊 Stress distribution:")
        for i, (label, count) in enumerate(zip(stress_labels, counts)):
            percentage = count / len(labels) * 100
            print(f"      • {label}: {count} samples ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return
    
    # Test different model architectures
    print("\n3. 🤖 Testing Different Model Architectures...")
    
    model_types = ["simple", "time_series", "bilstm"]
    model_results = {}
    
    for model_type in model_types:
        print(f"\n   🔄 Testing {model_type.upper()} model...")
        try:
            # Initialize framework with different model
            test_framework = ResponsibleAIFramework(model_type=model_type, privacy_epsilon=1.0)
            
            # Prepare data based on model type
            if model_type == "simple":
                train_features, train_labels = features, labels
                epochs = 3
            elif model_type == "time_series":
                # Use time series data
                train_features, train_labels = get_sample_time_series_for_demo()
                epochs = 3
            else:  # bilstm
                train_features, train_labels = features, labels
                epochs = 3
            
            # Train model
            result = test_framework.train_with_carbon_tracking(
                train_features, train_labels, epochs=epochs, use_privacy=True
            )
            
            if 'error' not in result:
                model_results[model_type] = result
                print(f"      ✅ {model_type.upper()} training completed")
                print(f"      📈 Accuracy: {result['accuracy']:.1%}")
                print(f"      🌱 Carbon: {result['carbon_emissions']['co2_kg']:.6f} kg CO2")
            else:
                print(f"      ❌ {model_type.upper()} training failed: {result['error']}")
                
        except Exception as e:
            print(f"      ⚠️  {model_type.upper()} model test failed: {e}")
    
    # Compare model performance
    if model_results:
        print(f"\n   📊 Model Performance Comparison:")
        for model_type, result in model_results.items():
            print(f"      • {model_type.upper()}: {result['accuracy']:.1%} accuracy, "
                  f"{result['carbon_emissions']['co2_kg']:.6f} kg CO2")
    
    # Train main model with carbon tracking
    print("\n4. 🌱 Training Main Model with Carbon Tracking...")
    try:
        training_result = framework.train_with_carbon_tracking(
            features, labels, epochs=10, use_privacy=True
        )
        
        if 'error' in training_result:
            print(f"   ❌ Training failed: {training_result['error']}")
            return
        
        print(f"   ✅ Training completed successfully!")
        print(f"   📈 Final accuracy: {training_result['accuracy']:.1%}")
        print(f"   🌱 Carbon emissions: {training_result['carbon_emissions']['co2_kg']:.6f} kg CO2")
        print(f"   ⚡ Energy consumption: {training_result['carbon_emissions']['energy_kwh']:.6f} kWh")
        print(f"   ⏱️  Training duration: {training_result['carbon_emissions'].get('duration_seconds', 0):.1f} seconds")
        print(f"   🔒 Privacy protection: {'Applied' if training_result['privacy_used'] else 'Not used'}")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return
    
    # Make predictions with SHAP explanations
    print("\n5. 🔍 Making Predictions with SHAP Explanations...")
    
    # Test multiple scenarios
    test_scenarios = [
        {"name": "Young Active Person", "data": [25, 175, 65, 2], "description": "25yo, 175cm, 65kg, high activity"},
        {"name": "Middle-aged Sedentary", "data": [45, 170, 80, 0], "description": "45yo, 170cm, 80kg, no activity"},
        {"name": "Senior Moderate", "data": [65, 165, 75, 1], "description": "65yo, 165cm, 75kg, moderate activity"},
    ]
    
    for scenario in test_scenarios:
        print(f"\n   🎯 Testing: {scenario['name']}")
        print(f"      📝 Profile: {scenario['description']}")
        
        try:
            test_input = np.array([scenario['data']])
            prediction = framework.predict_with_explanation(test_input)
            
            if 'error' in prediction:
                print(f"      ❌ Prediction failed: {prediction['error']}")
                continue
            
            print(f"      ✅ Predicted stress: {prediction['predicted_stress']}")
            print(f"      📊 Confidence: {prediction['confidence']:.1%}")
            
            # Show probability distribution
            if 'probabilities' in prediction:
                probs = prediction['probabilities']
                print(f"      📈 Probabilities: Low={probs['low']:.1%}, "
                      f"Medium={probs['medium']:.1%}, High={probs['high']:.1%}")
            
            # Show top 3 SHAP features
            if 'explanation' in prediction and 'feature_importance' in prediction['explanation']:
                print(f"      🔍 Top 3 influential features:")
                for i, feat in enumerate(prediction['explanation']['feature_importance'][:3]):
                    impact_icon = "↗️" if feat['impact'] == 'positive' else "↘️"
                    print(f"         {i+1}. {feat['feature']}: {feat['shap_value']:.3f} {impact_icon}")
            
        except Exception as e:
            print(f"      ❌ Prediction failed: {e}")
    
    # Federated learning simulation
    print("\n6. 🌐 Federated Learning Across Healthcare Institutions...")
    try:
        print("   🏥 Simulating 3 healthcare institutions...")
        institution_data = create_sample_healthcare_data(n_institutions=3, samples_per_institution=40)
        
        print(f"   📊 Institution data prepared:")
        for i, (inst_features, inst_labels) in enumerate(institution_data):
            stress_dist = np.bincount(inst_labels, minlength=3)
            print(f"      • Institution {i+1}: {len(inst_features)} patients, "
                  f"stress distribution: {stress_dist[0]}/{stress_dist[1]}/{stress_dist[2]}")
        
        federated_result = framework.federated_learning(
            institution_data, rounds=3
        )
        
        if 'error' not in federated_result:
            print(f"   ✅ Federated learning completed!")
            print(f"   🤝 Institutions: {len(institution_data)}")
            print(f"   🔄 Training rounds: 3")
            print(f"   🔒 Privacy protected: {federated_result['privacy_protected']}")
            print(f"   🌱 Carbon emissions: {federated_result['carbon_emissions']['co2_kg']:.6f} kg CO2")
            print(f"   📈 Final model updated with federated knowledge")
        else:
            print(f"   ❌ Federated learning failed: {federated_result['error']}")
            
    except Exception as e:
        print(f"   ❌ Federated learning failed: {e}")
    
    # Privacy analysis
    print("\n7. 🔒 Privacy Analysis...")
    try:
        # Test privacy features
        print("   🔍 Analyzing privacy protection mechanisms...")
        
        # Test differential privacy
        original_features = features[:20]
        private_features = framework.privacy_layer.add_differential_privacy_noise(
            original_features, sensitivity=1.0
        )
        
        noise_level = np.mean(np.abs(private_features - original_features))
        print(f"   📊 Differential privacy noise level: {noise_level:.4f}")
        
        # Test k-anonymity
        k_anon_features = framework.privacy_layer.k_anonymize_features(original_features, k=5)
        print(f"   🔐 K-anonymity applied with k=5")
        
        # Privacy budget calculation
        num_queries = len(framework.explanation_history) if hasattr(framework, 'explanation_history') else 0
        privacy_budget = framework.privacy_layer.compute_privacy_budget(num_queries)
        print(f"   💰 Privacy budget consumed: {privacy_budget:.4f}")
        
        # Generate privacy report
        privacy_report = framework.privacy_layer.get_privacy_report()
        print(f"   📋 Privacy level: {privacy_report['privacy_level']}")
        print(f"   🔒 Privacy epsilon: {privacy_report['epsilon']}")
        
    except Exception as e:
        print(f"   ❌ Privacy analysis failed: {e}")
    
    # Carbon footprint analysis
    print("\n8. 🌍 Carbon Footprint Analysis...")
    try:
        print("   🔬 Analyzing environmental impact...")
        
        # Calculate total training emissions
        total_training_co2 = sum(
            result.get('carbon_emissions', {}).get('co2_kg', 0) 
            for result in [training_result] + list(model_results.values())
        )
        
        total_training_energy = sum(
            result.get('carbon_emissions', {}).get('energy_kwh', 0) 
            for result in [training_result] + list(model_results.values())
        )
        
        print(f"   📊 Total training emissions: {total_training_co2:.6f} kg CO2")
        print(f"   ⚡ Total energy consumed: {total_training_energy:.6f} kWh")
        
        # Estimate carbon offset
        # 1 tree absorbs ~22kg CO2 per year
        trees_needed = total_training_co2 / 22
        print(f"   🌳 Trees needed to offset: {trees_needed:.8f} trees")
        
        # Healthcare benefits estimation
        patients_helped = len(features) * 10  # Estimate 10x more patients helped
        co2_per_patient = total_training_co2 / patients_helped
        print(f"   👥 Estimated patients helped: {patients_helped}")
        print(f"   💚 CO2 per patient helped: {co2_per_patient:.8f} kg")
        
    except Exception as e:
        print(f"   ❌ Carbon analysis failed: {e}")
    
    # Final framework status
    print("\n9. 📋 Final Framework Status...")
    try:
        final_status = framework.get_status()
        print(f"   Model trained: {'✅ Yes' if final_status['model_trained'] else '❌ No'}")
        print(f"   Training sessions: {final_status['training_history']}")
        print(f"   Explanations generated: {final_status['explanations_generated']}")
        print(f"   Carbon tracking: {'🌱 Active' if final_status['carbon_tracking_active'] else '❌ Inactive'}")
        print(f"   Privacy epsilon: 🔒 {final_status['privacy_epsilon']}")
        print(f"   SHAP explainer: {'🔍 Ready' if final_status['shap_explainer_initialized'] else '⚠️ Not ready'}")
        print(f"   Federated learning: {'🌐 Ready' if final_status['flower_fl_initialized'] else '⚠️ Not ready'}")
        
    except Exception as e:
        print(f"   ⚠️  Could not get final status: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Stress Detection Responsible AI Framework Demo Complete!")
    print("\n🌟 Key Features Demonstrated:")
    print("   ✅ Multiple neural network architectures (Simple, LSTM, BiLSTM)")
    print("   ✅ Real-time carbon footprint tracking during training")
    print("   ✅ SHAP explainability for medical decision support")
    print("   ✅ Federated learning across healthcare institutions")
    print("   ✅ Privacy preservation with differential privacy")
    print("   ✅ Comprehensive responsible AI metrics and reporting")
    
    print("\n🚀 Next Steps:")
    print("   • Run 'streamlit run main_app.py' for interactive web interface")
    print("   • Try 'python street_light_demo.py' for IoT smart city demo")
    print("   • Run 'python test_integration.py' to verify all components")
    print("   • Explore 'src/stress_detection/' for framework internals")
    
    print("\n🏥 Healthcare Applications:")
    print("   • Real-time stress monitoring from wearable devices")
    print("   • Privacy-safe patient data analysis")
    print("   • Multi-hospital collaborative research")
    print("   • Explainable AI for medical professionals")
    
    print(f"\n⏰ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return framework


if __name__ == "__main__":
    try:
        framework = main()
        print(f"\n📋 Framework object available for further exploration")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
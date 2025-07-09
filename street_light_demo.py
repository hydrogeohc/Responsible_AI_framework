"""
Street Light IoT Responsible AI Framework Demo
Demonstrates integration of carbon tracking, SHAP explainability, 
Flower federated learning, and PySyft privacy preservation.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import organized modules
from src.street_light import StreetLightResponsibleAI, StreetLightDataProcessor, create_sample_district_data
from config import get_config


def main():
    """Main demonstration function."""
    print("🏙️  Street Light IoT Responsible AI Framework Demo")
    print("=" * 60)
    
    # Initialize the framework with configuration
    print("\n1. Initializing Responsible AI Framework...")
    config = get_config('dev')  # Use development configuration
    framework = StreetLightResponsibleAI(config=config)
    
    # Load street light data
    print("\n2. Loading Street Light Data...")
    data_processor = StreetLightDataProcessor()
    
    # Try to load real data, fallback to synthetic
    try:
        current_dir = os.path.dirname(__file__)
        led_data = data_processor.load_street_light_data(
            os.path.join(current_dir, "smart_city_light/bsl-cumulative-number-of-streetlights-converted-to-led.csv")
        )
        outage_data = data_processor.load_street_light_data(
            os.path.join(current_dir, "smart_city_light/bsl-percent-of-streetlight-outages-repaired-within-10-business-days.csv")
        )
        
        if led_data.empty or outage_data.empty:
            raise Exception("Data files not found or empty")
            
        print(f"   ✓ Loaded {len(led_data)} LED conversion records")
        print(f"   ✓ Loaded {len(outage_data)} outage repair records")
        
        # Use LED data as primary dataset
        street_light_data = led_data
        
    except Exception as e:
        print(f"   ⚠️  Could not load real data: {e}")
        print("   📊 Generating synthetic street light data...")
        
        # Generate synthetic data
        street_light_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100, freq='M'),
            'Cumulative # of streetlights converted to LED': np.random.randint(50000, 200000, 100),
            '% outages repaired within 10 business days': np.random.randint(80, 99, 100)
        })
        
        print(f"   ✓ Generated {len(street_light_data)} synthetic records")
    
    # Display framework status
    print("\n3. Framework Status:")
    status = framework.get_framework_status()
    print(f"   • Privacy epsilon: {status['privacy']['epsilon']}")
    print(f"   • Carbon tracking: {'Active' if status['carbon_tracking']['carbon_tracker_active'] else 'Inactive'}")
    print(f"   • Differential privacy: {'Enabled' if status['privacy']['differential_privacy_enabled'] else 'Disabled'}")
    print(f"   • Federated learning: {'Ready' if status['federated_learning']['multi_district_ready'] else 'Not Ready'}")
    
    # Train performance model with carbon tracking
    print("\n4. Training Performance Model with Carbon Tracking...")
    performance_result = framework.train_performance_model(
        street_light_data, 
        epochs=10, 
        use_privacy=True
    )
    
    if 'error' not in performance_result:
        print(f"   ✓ Training completed with {performance_result['accuracy']:.1%} accuracy")
        print(f"   📊 Carbon emissions: {performance_result['carbon_emissions']['co2_kg']:.4f} kg CO2")
        print(f"   ⚡ Energy consumption: {performance_result['carbon_emissions']['energy_kwh']:.4f} kWh")
    else:
        print(f"   ❌ Training failed: {performance_result['error']}")
    
    # Train carbon model
    print("\n5. Training Carbon Footprint Model...")
    carbon_result = framework.train_carbon_model(
        street_light_data,
        epochs=8,
        use_privacy=True
    )
    
    if 'error' not in carbon_result:
        print(f"   ✓ Carbon model trained with MAE: {carbon_result['mae']:.4f}")
        print(f"   📊 Training emissions: {carbon_result['carbon_emissions']['co2_kg']:.4f} kg CO2")
    else:
        print(f"   ❌ Carbon training failed: {carbon_result['error']}")
    
    # Make predictions with SHAP explanations
    print("\n6. Making Predictions with SHAP Explanations...")
    
    # Performance prediction
    perf_prediction = framework.predict_with_explanation(
        street_light_data.head(1), 
        model_type="performance"
    )
    
    if 'error' not in perf_prediction:
        print(f"   🔍 Performance Status: {perf_prediction['predicted_status']}")
        print(f"   📊 Confidence: {perf_prediction['confidence']:.1%}")
        print(f"   ⚡ Energy consumption: {perf_prediction['energy_consumption_kwh']:.2f} kWh")
        print(f"   🌱 Carbon footprint: {perf_prediction['carbon_footprint_kg']:.4f} kg CO2")
        
        # Display top 3 important features
        if 'explanation' in perf_prediction and 'feature_importance' in perf_prediction['explanation']:
            print("   🔍 Top 3 Important Features:")
            for i, feat in enumerate(perf_prediction['explanation']['feature_importance'][:3]):
                print(f"      {i+1}. {feat['feature']}: {feat['importance']:.4f} ({feat['impact']})")
    
    # Carbon prediction
    carbon_prediction = framework.predict_with_explanation(
        street_light_data.head(1),
        model_type="carbon"
    )
    
    if 'error' not in carbon_prediction:
        print(f"   🌱 Carbon Level: {carbon_prediction['carbon_level']}")
        print(f"   📊 Predicted CO2: {carbon_prediction['predicted_carbon_kg']:.4f} kg")
    
    # Federated learning simulation
    print("\n7. Federated Learning Simulation...")
    print("   🏙️  Simulating 3 city districts...")
    
    district_data = create_sample_district_data(n_districts=3, samples_per_district=50)
    
    federated_result = framework.federated_learning_simulation(
        district_data,
        rounds=3,
        model_type="performance"
    )
    
    if 'error' not in federated_result:
        print(f"   ✓ Federated training completed across {federated_result['districts']} districts")
        print(f"   📊 Training rounds: {federated_result['federated_training']['rounds']}")
        print(f"   🔒 Privacy protected: {federated_result['privacy_protected']}")
        print(f"   🌱 Carbon emissions: {federated_result['carbon_emissions']['co2_kg']:.4f} kg CO2")
        
        # Display round results
        if 'results' in federated_result['federated_training']:
            for round_result in federated_result['federated_training']['results']:
                print(f"      Round {round_result['round']}: "
                      f"Loss={round_result['loss']:.4f}, "
                      f"Accuracy={round_result['accuracy']:.1%}")
    else:
        print(f"   ❌ Federated learning failed: {federated_result['error']}")
    
    # Privacy analysis
    print("\n8. Privacy Analysis...")
    privacy_analysis = framework.privacy_analysis(street_light_data)
    
    if 'error' not in privacy_analysis:
        print(f"   🔒 Privacy budget consumed: {privacy_analysis['privacy_budget_consumed']:.4f}")
        print(f"   📊 K-anonymity applied: {privacy_analysis['k_anonymized_samples']} samples")
        print(f"   🔐 Homomorphic encryption: {privacy_analysis['encryption_status']['encrypted']}")
        print(f"   🛡️  Privacy level: {privacy_analysis['privacy_report']['privacy_level']}")
    else:
        print(f"   ❌ Privacy analysis failed: {privacy_analysis['error']}")
    
    # Generate carbon report
    print("\n9. Carbon Footprint Report...")
    carbon_report = framework.get_carbon_report()
    
    if 'error' not in carbon_report:
        print(f"   📊 Training emissions: {carbon_report['training_emissions']['total_co2_kg']:.4f} kg CO2")
        print(f"   💡 LED lights estimated: {carbon_report['led_impact']['estimated_led_lights']:,}")
        print(f"   🌱 Annual carbon savings: {carbon_report['led_impact']['annual_carbon_savings_kg']:.0f} kg CO2")
        print(f"   ⚡ Annual energy savings: {carbon_report['led_impact']['annual_energy_savings_kwh']:.0f} kWh")
        print(f"   🌍 Net carbon impact: {carbon_report['sustainability_metrics']['net_carbon_impact_kg']:.0f} kg CO2")
        print(f"   ✅ Carbon positive: {carbon_report['sustainability_metrics']['carbon_positive']}")
    else:
        print(f"   ❌ Carbon report failed: {carbon_report['error']}")
    
    # Final framework status
    print("\n10. Final Framework Status:")
    final_status = framework.get_framework_status()
    print(f"   📊 Training sessions: {final_status['history']['training_sessions']}")
    print(f"   🔍 Predictions made: {final_status['history']['predictions_made']}")
    print(f"   🤖 Models trained: {sum(final_status['models'].values())}")
    print(f"   💡 Explainers ready: {sum(final_status['explainers'].values())}")
    
    print("\n" + "=" * 60)
    print("🎉 Street Light IoT Responsible AI Framework Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("✓ Carbon footprint tracking during model training")
    print("✓ SHAP explainability for maintenance predictions")
    print("✓ Flower federated learning across city districts")
    print("✓ Privacy preservation with differential privacy")
    print("✓ Integrated sustainability metrics")
    print("✓ Multi-model architecture (performance, carbon, LED)")
    
    return framework


if __name__ == "__main__":
    try:
        framework = main()
        print(f"\n📋 Framework object available as 'framework' variable")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
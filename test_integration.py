"""
Integration Test for Responsible AI Framework
Tests both stress detection and street light systems.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_stress_detection():
    """Test the original stress detection framework."""
    print("Testing Stress Detection Framework...")
    
    try:
        from src.stress_detection.core.framework import ResponsibleAIFramework
        from src.stress_detection.data.data_utils import get_sample_data_for_demo
        
        # Initialize framework
        framework = ResponsibleAIFramework(model_type="simple", privacy_epsilon=2.0)
        
        # Get sample data
        features, labels = get_sample_data_for_demo()
        
        # Test training (minimal epochs for testing)
        print("  ✓ Training stress detection model...")
        result = framework.train_with_carbon_tracking(features, labels, epochs=2, use_privacy=True)
        
        # Test prediction
        print("  ✓ Making prediction with explanation...")
        prediction = framework.predict_with_explanation(features[:1])
        
        print(f"  ✓ Stress Detection Test Passed - Accuracy: {result['accuracy']:.2%}")
        return True
        
    except Exception as e:
        print(f"  ❌ Stress Detection Test Failed: {e}")
        return False

def test_street_light_system():
    """Test the street light system."""
    print("Testing Street Light System...")
    
    try:
        from src.street_light import StreetLightResponsibleAI, StreetLightDataProcessor, create_sample_district_data
        from config import get_config
        
        # Initialize framework
        config = get_config('test')  # Use test configuration
        framework = StreetLightResponsibleAI(config=config)
        
        # Create sample data
        data_processor = StreetLightDataProcessor()
        street_light_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=20, freq='M'),
            'Cumulative # of streetlights converted to LED': np.random.randint(1000, 200000, 20),
            '% outages repaired within 10 business days': np.random.randint(75, 99, 20)
        })
        
        # Test performance model training
        print("  ✓ Training performance model...")
        result = framework.train_performance_model(street_light_data, epochs=2, use_privacy=True)
        
        # Test prediction with explanation
        print("  ✓ Making prediction with explanation...")
        prediction = framework.predict_with_explanation(street_light_data.head(1), model_type="performance")
        
        # Test federated learning
        print("  ✓ Testing federated learning simulation...")
        district_data = create_sample_district_data(n_districts=2, samples_per_district=10)
        fed_result = framework.federated_learning_simulation(district_data, rounds=1, model_type="performance")
        
        # Test privacy analysis
        print("  ✓ Testing privacy analysis...")
        privacy_report = framework.privacy_analysis(street_light_data)
        
        # Test carbon report
        print("  ✓ Testing carbon report...")
        carbon_report = framework.get_carbon_report()
        
        print(f"  ✓ Street Light System Test Passed - Accuracy: {result['accuracy']:.2%}")
        return True
        
    except Exception as e:
        print(f"  ❌ Street Light System Test Failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("Testing Configuration System...")
    
    try:
        from config import get_config, validate_config
        
        # Test different configurations
        dev_config = get_config('dev')
        prod_config = get_config('prod')
        test_config = get_config('test')
        
        # Validate configurations
        assert validate_config(dev_config), "Dev config validation failed"
        assert validate_config(prod_config), "Prod config validation failed"
        assert validate_config(test_config), "Test config validation failed"
        
        # Test different privacy settings
        assert dev_config['privacy_epsilon'] == 5.0
        assert prod_config['privacy_epsilon'] == 0.5
        assert test_config['privacy_epsilon'] == 10.0
        
        print("  ✓ Configuration System Test Passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration System Test Failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 Running Integration Tests for Responsible AI Framework")
    print("=" * 60)
    
    tests = [
        test_configuration,
        test_stress_detection,
        test_street_light_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
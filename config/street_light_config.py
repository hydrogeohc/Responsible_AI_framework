"""
Street Light Configuration
Configuration settings for the street light responsible AI framework.
"""

import os
from typing import Dict, Any

# Base configuration
BASE_CONFIG = {
    # Privacy settings
    'privacy_epsilon': 1.0,
    'k_anonymity': 5,
    
    # Carbon tracking
    'carbon_log_dir': './carbon_logs',
    'carbon_tracking_enabled': True,
    
    # Model parameters
    'model_params': {
        'performance': {
            'INPUT_DIM': 8,
            'HIDDEN_DIM': 128,
            'OUTPUT_DIM': 3,
            'DROPOUT_RATE': 0.2
        },
        'carbon': {
            'INPUT_DIM': 10,
            'HIDDEN_DIM': 96,
            'DROPOUT_RATE': 0.2
        },
        'led': {
            'INPUT_DIM': 6,
            'HIDDEN_DIM': 64,
            'SEQUENCE_LENGTH': 12,
            'OUTPUT_DIM': 2,
            'NUM_HEADS': 4
        }
    },
    
    # Training settings
    'training': {
        'LEARNING_RATE': 0.001,
        'BATCH_SIZE': 32,
        'MAX_EPOCHS': 100,
        'EARLY_STOPPING_PATIENCE': 10,
        'VALIDATION_SPLIT': 0.2
    },
    
    # Federated learning
    'federated_learning': {
        'DEFAULT_ROUNDS': 3,
        'LOCAL_EPOCHS': 5,
        'MIN_CLIENTS': 2,
        'MAX_CLIENTS': 10
    },
    
    # Data settings
    'data': {
        'data_dir': 'smart_city_light',
        'min_samples': 10,
        'max_missing_percentage': 0.3
    },
    
    # Explainability
    'explainability': {
        'shap_samples': 100,
        'background_samples': 50,
        'top_features': 5
    }
}

# Development configuration
DEV_CONFIG = BASE_CONFIG.copy()
DEV_CONFIG.update({
    'privacy_epsilon': 5.0,  # Less strict privacy for development
    'training': {
        **BASE_CONFIG['training'],
        'MAX_EPOCHS': 10,  # Faster training for development
        'LEARNING_RATE': 0.01
    },
    'carbon_tracking_enabled': False,  # Disable carbon tracking for faster development
})

# Production configuration
PROD_CONFIG = BASE_CONFIG.copy()
PROD_CONFIG.update({
    'privacy_epsilon': 0.5,  # Stricter privacy for production
    'training': {
        **BASE_CONFIG['training'],
        'MAX_EPOCHS': 200,  # More thorough training
        'LEARNING_RATE': 0.0001
    },
    'carbon_tracking_enabled': True,
    'federated_learning': {
        **BASE_CONFIG['federated_learning'],
        'DEFAULT_ROUNDS': 10,  # More federated rounds for production
        'LOCAL_EPOCHS': 10
    }
})

# Test configuration
TEST_CONFIG = BASE_CONFIG.copy()
TEST_CONFIG.update({
    'privacy_epsilon': 10.0,  # Very relaxed privacy for testing
    'training': {
        **BASE_CONFIG['training'],
        'MAX_EPOCHS': 2,  # Minimal training for testing
        'LEARNING_RATE': 0.1
    },
    'carbon_tracking_enabled': False,
    'data': {
        **BASE_CONFIG['data'],
        'min_samples': 5
    }
})

# Configuration selector
CONFIGURATIONS = {
    'base': BASE_CONFIG,
    'dev': DEV_CONFIG,
    'prod': PROD_CONFIG,
    'test': TEST_CONFIG
}

def get_config(environment: str = 'base') -> Dict[str, Any]:
    """
    Get configuration for specified environment.
    
    Args:
        environment: Environment name ('base', 'dev', 'prod', 'test')
        
    Returns:
        Configuration dictionary
    """
    if environment not in CONFIGURATIONS:
        raise ValueError(f"Unknown environment: {environment}. Available: {list(CONFIGURATIONS.keys())}")
    
    config = CONFIGURATIONS[environment].copy()
    
    # Override with environment variables if available
    if 'STREET_LIGHT_PRIVACY_EPSILON' in os.environ:
        config['privacy_epsilon'] = float(os.environ['STREET_LIGHT_PRIVACY_EPSILON'])
    
    if 'STREET_LIGHT_CARBON_LOG_DIR' in os.environ:
        config['carbon_log_dir'] = os.environ['STREET_LIGHT_CARBON_LOG_DIR']
    
    if 'STREET_LIGHT_MAX_EPOCHS' in os.environ:
        config['training']['MAX_EPOCHS'] = int(os.environ['STREET_LIGHT_MAX_EPOCHS'])
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['privacy_epsilon', 'model_params', 'training', 'federated_learning']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required configuration key: {key}")
            return False
    
    # Validate privacy epsilon
    if not 0 < config['privacy_epsilon'] <= 20:
        print(f"Invalid privacy_epsilon: {config['privacy_epsilon']} (must be between 0 and 20)")
        return False
    
    # Validate model parameters
    for model_type in ['performance', 'carbon', 'led']:
        if model_type not in config['model_params']:
            print(f"Missing model parameters for: {model_type}")
            return False
    
    return True
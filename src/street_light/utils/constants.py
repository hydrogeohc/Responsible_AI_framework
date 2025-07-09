"""
Street Light Constants
Constants and configuration values for the street light framework.
"""

from typing import Dict, List

# Model Constants
STREET_LIGHT_CONSTANTS = {
    # Model Architecture
    'PERFORMANCE_MODEL': {
        'INPUT_DIM': 8,
        'HIDDEN_DIM': 128,
        'OUTPUT_DIM': 3,
        'DROPOUT_RATE': 0.2
    },
    
    'CARBON_MODEL': {
        'INPUT_DIM': 10,
        'HIDDEN_DIM': 96,
        'DROPOUT_RATE': 0.2
    },
    
    'LED_MODEL': {
        'INPUT_DIM': 6,
        'HIDDEN_DIM': 64,
        'SEQUENCE_LENGTH': 12,
        'OUTPUT_DIM': 2,
        'NUM_HEADS': 4
    },
    
    # Training Parameters
    'TRAINING': {
        'LEARNING_RATE': 0.001,
        'BATCH_SIZE': 32,
        'MAX_EPOCHS': 100,
        'EARLY_STOPPING_PATIENCE': 10,
        'VALIDATION_SPLIT': 0.2
    },
    
    # Privacy Parameters
    'PRIVACY': {
        'DEFAULT_EPSILON': 1.0,
        'K_ANONYMITY': 5,
        'NOISE_SCALE_FACTOR': 1.0,
        'MIN_EPSILON': 0.1,
        'MAX_EPSILON': 10.0
    },
    
    # Carbon Tracking
    'CARBON': {
        'DEFAULT_LOG_DIR': './carbon_logs',
        'CARBON_INTENSITY_KG_PER_KWH': 0.5,
        'TRACKING_ENABLED': True,
        'COMPONENTS': 'all'  # 'all', 'gpu', 'cpu'
    },
    
    # Energy Calculations
    'ENERGY': {
        'TRADITIONAL_LIGHT_POWER_W': 150.0,
        'LED_LIGHT_POWER_W': 75.0,
        'OPERATING_HOURS_PER_DAY': 12.0,
        'ELECTRICITY_COST_PER_KWH': 0.12,
        'ENERGY_SAVINGS_FACTOR': 0.5
    },
    
    # Street Light Categories
    'PERFORMANCE_CLASSES': {
        0: 'Operational',
        1: 'Needs Maintenance', 
        2: 'Critical Failure'
    },
    
    'CARBON_LEVELS': {
        'LOW': (0.0, 0.5),
        'MEDIUM': (0.5, 1.0),
        'HIGH': (1.0, float('inf'))
    },
    
    # Feature Names
    'FEATURE_NAMES': {
        'PERFORMANCE': [
            'LED Count',
            'Operational %',
            'Repair Rate',
            'Energy Usage',
            'Age (years)',
            'Location Density',
            'Weather Factor',
            'Usage Pattern'
        ],
        
        'CARBON': [
            'LED Count',
            'Operational %',
            'Repair Rate',
            'Energy Usage',
            'Age',
            'Location Density',
            'Weather',
            'Usage Pattern',
            'Base Energy',
            'Carbon Intensity'
        ],
        
        'LED': [
            'Current LED Count',
            'Conversion Rate',
            'Energy Efficiency',
            'Maintenance Cost',
            'Location Priority',
            'Budget Allocation'
        ]
    },
    
    # Data Validation
    'DATA_VALIDATION': {
        'MIN_SAMPLES': 10,
        'MAX_MISSING_PERCENTAGE': 0.3,
        'REQUIRED_COLUMNS': {
            'led_conversion': [
                'Date Name',
                'Date',
                'Cumulative # of streetlights converted to LED'
            ],
            'outage_repair': [
                'Date Name',
                'Date Value',
                '% outages repaired within 10 business days'
            ]
        }
    },
    
    # Federated Learning
    'FEDERATED_LEARNING': {
        'MIN_CLIENTS': 2,
        'MAX_CLIENTS': 10,
        'DEFAULT_ROUNDS': 3,
        'LOCAL_EPOCHS': 5,
        'AGGREGATION_METHOD': 'fedavg'
    },
    
    # Explainability
    'EXPLAINABILITY': {
        'SHAP_SAMPLES': 100,
        'EXPLAINER_TYPE': 'kernel',  # 'kernel', 'deep', 'gradient'
        'TOP_FEATURES': 5,
        'BACKGROUND_SAMPLES': 50
    }
}

# File Patterns
SUPPORTED_FILE_PATTERNS = {
    'CSV': ['.csv'],
    'JSON': ['.json'],
    'PARQUET': ['.parquet'],
    'EXCEL': ['.xlsx', '.xls']
}

# Default Configuration
DEFAULT_CONFIG = {
    'privacy_epsilon': STREET_LIGHT_CONSTANTS['PRIVACY']['DEFAULT_EPSILON'],
    'carbon_log_dir': STREET_LIGHT_CONSTANTS['CARBON']['DEFAULT_LOG_DIR'],
    'model_params': {
        'performance': STREET_LIGHT_CONSTANTS['PERFORMANCE_MODEL'],
        'carbon': STREET_LIGHT_CONSTANTS['CARBON_MODEL'],
        'led': STREET_LIGHT_CONSTANTS['LED_MODEL']
    },
    'training': STREET_LIGHT_CONSTANTS['TRAINING'],
    'federated_learning': STREET_LIGHT_CONSTANTS['FEDERATED_LEARNING']
}

# Error Messages
ERROR_MESSAGES = {
    'MISSING_DATA': "Required data file not found: {}",
    'INVALID_CONFIG': "Invalid configuration: {}",
    'MODEL_NOT_TRAINED': "Model {} has not been trained yet",
    'INSUFFICIENT_DATA': "Insufficient data for training: need at least {} samples",
    'PRIVACY_VIOLATION': "Privacy budget exceeded: current {}, limit {}",
    'CARBON_TRACKING_FAILED': "Carbon tracking initialization failed: {}",
    'FEDERATED_LEARNING_FAILED': "Federated learning setup failed: {}"
}

# Success Messages
SUCCESS_MESSAGES = {
    'MODEL_TRAINED': "Successfully trained {} model with accuracy: {:.2%}",
    'PREDICTION_COMPLETE': "Prediction completed for {} samples",
    'CARBON_TRACKED': "Carbon tracking completed: {:.4f} kg CO2 emitted",
    'PRIVACY_PRESERVED': "Privacy protection applied with epsilon: {}",
    'FEDERATED_COMPLETE': "Federated learning completed across {} clients"
}
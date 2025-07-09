"""
Main Streamlit application for the Responsible AI Framework.
Clean, organized interface for stress detection with carbon tracking,
privacy preservation, federated learning, and model interpretability.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import framework components
from core.framework import ResponsibleAIFramework
from data.data_utils import get_sample_data_for_demo
from data.time_series_utils import get_sample_time_series_for_demo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Responsible AI Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .warning-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .info-badge {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'framework' not in st.session_state:
    st.session_state.framework = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Responsible AI Framework</h1>
    <p>Comprehensive stress detection with carbon tracking, privacy preservation, and explainable AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Architecture",
    ["simple", "time_series", "bilstm"],
    help="Choose the neural network architecture for stress detection"
)

# Privacy settings
st.sidebar.subheader("🔒 Privacy Settings")
privacy_epsilon = st.sidebar.slider(
    "Privacy Budget (ε)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Lower values = more privacy, higher values = better accuracy"
)

# Feature toggles
use_privacy = st.sidebar.checkbox("Enable Privacy Protection", value=True)
use_carbon_tracking = st.sidebar.checkbox("Enable Carbon Tracking", value=True)
use_explanations = st.sidebar.checkbox("Enable SHAP Explanations", value=True)

# Initialize framework
if (st.session_state.framework is None or 
    st.session_state.framework.model_type != model_type or
    st.session_state.framework.privacy_layer.epsilon != privacy_epsilon):
    
    try:
        with st.spinner("Initializing framework..."):
            st.session_state.framework = ResponsibleAIFramework(
                model_type=model_type,
                privacy_epsilon=privacy_epsilon
            )
            st.session_state.training_complete = False
        st.sidebar.success("✅ Framework ready!")
    except Exception as e:
        st.sidebar.error(f"❌ Initialization failed: {e}")
        st.stop()

framework = st.session_state.framework

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Training", 
    "🔮 Prediction", 
    "📊 Analytics", 
    "⚙️ System Status"
])

# Tab 1: Training
with tab1:
    st.header("Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        # Training parameters
        epochs = st.slider("Training Epochs", 5, 20, 10)
        
        # Training mode
        training_mode = st.radio(
            "Training Mode",
            ["Single Node", "Federated Learning"]
        )
        
        if training_mode == "Federated Learning":
            num_clients = st.slider("Number of Clients", 2, 5, 3)
            fed_rounds = st.slider("Federated Rounds", 1, 5, 3)
        
        # Start training
        if st.button("🚀 Start Training", type="primary"):
            try:
                if training_mode == "Single Node":
                    # Single node training
                    with st.spinner("Training with carbon tracking..."):
                        features, labels = get_sample_data_for_demo()
                        
                        training_result = framework.train_with_carbon_tracking(
                            features, labels, 
                            epochs=epochs, 
                            use_privacy=use_privacy
                        )
                        
                        st.session_state.training_complete = True
                        st.success("✅ Training completed!")
                        
                        # Display results
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Accuracy", f"{training_result['accuracy']:.1%}")
                        with col_b:
                            st.metric("CO₂ Emissions", f"{training_result['carbon_emissions']['co2_kg']:.6f} kg")
                        with col_c:
                            st.metric("Energy Used", f"{training_result['carbon_emissions']['energy_kwh']:.6f} kWh")
                
                else:
                    # Federated learning
                    with st.spinner("Running federated learning..."):
                        client_data = []
                        for i in range(num_clients):
                            features, labels = get_sample_data_for_demo()
                            client_data.append((features, labels))
                        
                        fed_result = framework.federated_learning(client_data, rounds=fed_rounds)
                        
                        st.session_state.training_complete = True
                        st.success("✅ Federated learning completed!")
                        
                        # Display results
                        if fed_result['federated_training']['success']:
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Clients", fed_result['federated_training']['clients'])
                            with col_b:
                                st.metric("Rounds", fed_result['federated_training']['rounds'])
                            with col_c:
                                st.metric("Privacy Protected", "Yes" if fed_result['privacy_protected'] else "No")
                            
                            # Show progress
                            if 'results' in fed_result['federated_training']:
                                results_df = pd.DataFrame(fed_result['federated_training']['results'])
                                fig = px.line(results_df, x='round', y=['loss', 'accuracy'],
                                            title="Federated Learning Progress")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("❌ Federated learning failed")
                            
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                logger.error(f"Training error: {e}")
    
    with col2:
        st.subheader("Training Status")
        
        if st.session_state.training_complete:
            st.markdown('<span class="success-badge">✅ Model Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="warning-badge">⚠️ Not Trained</span>', unsafe_allow_html=True)
        
        # Framework status
        status = framework.get_status()
        st.markdown(f"""
        <div class="metric-card">
            <strong>Framework Status</strong><br>
            <b>Model:</b> {status['model_type']}<br>
            <b>Training Runs:</b> {status['training_history']}<br>
            <b>Carbon Tracking:</b> {status['carbon_tracking_active']}<br>
            <b>Privacy ε:</b> {status['privacy_epsilon']:.2f}<br>
            <b>SHAP Ready:</b> {status['shap_explainer_initialized']}
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Prediction
with tab2:
    st.header("Stress Level Prediction")
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train the model first")
    else:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Patient Input")
            
            # Input method
            input_method = st.radio(
                "Input Method",
                ["Manual Input", "Predefined Scenarios"]
            )
            
            if input_method == "Manual Input":
                # Manual input fields
                age = st.slider("Age", 18, 80, 35)
                height = st.slider("Height (cm)", 140, 220, 170)
                weight = st.slider("Weight (kg)", 40, 150, 70)
                activity = st.selectbox("Regular Physical Activity", ["No", "Yes"])
                
                activity_val = 1 if activity == "Yes" else 0
                input_data = np.array([[age, height, weight, activity_val]])
                
            else:
                # Predefined scenarios
                scenarios = {
                    "Healthy Young Adult": [25, 175, 70, 1],
                    "Sedentary Worker": [40, 170, 85, 0],
                    "Active Senior": [65, 165, 75, 1],
                    "Stressed Professional": [35, 172, 90, 0]
                }
                
                selected_scenario = st.selectbox("Select Scenario", list(scenarios.keys()))
                input_data = np.array([scenarios[selected_scenario]])
                
                # Display scenario
                st.info(f"Age: {input_data[0][0]}, Height: {input_data[0][1]}cm, Weight: {input_data[0][2]}kg, Activity: {'Yes' if input_data[0][3] else 'No'}")
            
            # Prediction button
            if st.button("🔮 Predict Stress Level", type="primary"):
                try:
                    with st.spinner("Generating prediction..."):
                        prediction_result = framework.predict_with_explanation(
                            input_data, 
                            generate_explanation=use_explanations
                        )
                        
                        # Store in history
                        st.session_state.prediction_history.append({
                            'input': input_data.tolist(),
                            'result': prediction_result,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Display prediction
                        st.success(f"**Predicted Stress Level: {prediction_result['predicted_stress']}**")
                        st.write(f"**Confidence: {prediction_result['confidence']:.1%}**")
                        
                        # Probability distribution
                        probs = prediction_result['probabilities']
                        prob_df = pd.DataFrame({
                            'Stress Level': ['Low', 'Medium', 'High'],
                            'Probability': [probs['low'], probs['medium'], probs['high']]
                        })
                        
                        fig = px.bar(prob_df, x='Stress Level', y='Probability',
                                   title="Stress Level Probability Distribution",
                                   color='Probability',
                                   color_continuous_scale='RdYlBu_r')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # SHAP explanation
                        if use_explanations and 'explanation' in prediction_result:
                            explanation = prediction_result['explanation']
                            if 'error' not in explanation and 'feature_importance' in explanation:
                                st.subheader("🔍 Feature Importance")
                                
                                importance_df = pd.DataFrame(explanation['feature_importance'])
                                fig = px.bar(importance_df, x='importance', y='feature',
                                           orientation='h',
                                           title="SHAP Feature Importance",
                                           color='impact',
                                           color_discrete_map={'positive': 'red', 'negative': 'blue'})
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Feature explanations
                                st.write("**Feature Impact Analysis:**")
                                for feat in importance_df.itertuples():
                                    impact_icon = "📈" if feat.impact == 'positive' else "📉"
                                    st.write(f"{impact_icon} **{feat.feature}**: {feat.shap_value:.3f}")
                            else:
                                st.info("Using gradient-based explanation fallback")
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    logger.error(f"Prediction error: {e}")
        
        with col2:
            st.subheader("Recent Predictions")
            
            if st.session_state.prediction_history:
                recent_predictions = st.session_state.prediction_history[-3:]
                
                for i, pred in enumerate(reversed(recent_predictions)):
                    with st.expander(f"Prediction {len(recent_predictions) - i}"):
                        result = pred['result']
                        st.write(f"**Result:** {result['predicted_stress']}")
                        st.write(f"**Confidence:** {result['confidence']:.1%}")
                        st.write(f"**Time:** {pred['timestamp'][:19]}")
            else:
                st.info("No predictions yet")

# Tab 3: Analytics
with tab3:
    st.header("Analytics Dashboard")
    
    if not st.session_state.prediction_history:
        st.info("No analytics data available. Make predictions to see analytics.")
    else:
        # Process prediction history
        predictions_df = pd.DataFrame([
            {
                'timestamp': pred['timestamp'],
                'predicted_stress': pred['result']['predicted_stress'],
                'confidence': pred['result']['confidence']
            }
            for pred in st.session_state.prediction_history
        ])
        
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(predictions_df))
        
        with col2:
            avg_confidence = predictions_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            high_stress_rate = (predictions_df['predicted_stress'] == 'High Stress').mean()
            st.metric("High Stress Rate", f"{high_stress_rate:.1%}")
        
        with col4:
            training_runs = len(framework.training_history)
            st.metric("Training Runs", training_runs)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Stress distribution
            stress_counts = predictions_df['predicted_stress'].value_counts()
            fig = px.pie(values=stress_counts.values, names=stress_counts.index,
                        title="Stress Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence over time
            fig = px.line(predictions_df, x='timestamp', y='confidence',
                         title="Prediction Confidence Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Carbon footprint analysis
        if framework.training_history:
            st.subheader("Carbon Footprint Analysis")
            
            carbon_data = []
            for i, training in enumerate(framework.training_history):
                carbon_data.append({
                    'run': i + 1,
                    'co2_kg': training['carbon_emissions']['co2_kg'],
                    'energy_kwh': training['carbon_emissions']['energy_kwh']
                })
            
            carbon_df = pd.DataFrame(carbon_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(carbon_df, x='run', y='co2_kg',
                           title="CO₂ Emissions by Training Run")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(carbon_df, x='run', y='energy_kwh',
                           title="Energy Consumption by Training Run")
                st.plotly_chart(fig, use_container_width=True)

# Tab 4: System Status
with tab4:
    st.header("System Status")
    
    status = framework.get_status()
    
    # System overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧠 Model Status</h4>
            <b>Type:</b> {status['model_type']}<br>
            <b>Trained:</b> {status['model_trained']}<br>
            <b>Training Runs:</b> {status['training_history']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔒 Privacy Status</h4>
            <b>Epsilon:</b> {status['privacy_epsilon']:.2f}<br>
            <b>Protection:</b> {'On' if use_privacy else 'Off'}<br>
            <b>Mechanism:</b> Laplace DP
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🌱 Carbon Tracking</h4>
            <b>Active:</b> {status['carbon_tracking_active']}<br>
            <b>Monitoring:</b> {'On' if use_carbon_tracking else 'Off'}<br>
            <b>Component:</b> CarbonTracker
        </div>
        """, unsafe_allow_html=True)
    
    # Component status
    st.subheader("Component Status")
    
    components = [
        ("🔍 SHAP Explainer", status['shap_explainer_initialized']),
        ("🌸 Flower FL", status['flower_fl_initialized']),
        ("🌱 Carbon Tracker", status['carbon_tracking_active']),
        ("🔒 Privacy Layer", True)
    ]
    
    for name, active in components:
        badge_class = "success-badge" if active else "warning-badge"
        status_text = "Active" if active else "Inactive"
        st.markdown(f'{name}: <span class="{badge_class}">{status_text}</span>', unsafe_allow_html=True)
    
    # Privacy report
    st.subheader("Privacy Report")
    privacy_report = framework.privacy_layer.get_privacy_report()
    
    st.json(privacy_report)

# Sidebar status
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Status")

status = framework.get_status()
st.sidebar.write(f"**Model:** {status['model_type']}")
st.sidebar.write(f"**Trained:** {'Yes' if status['model_trained'] else 'No'}")
st.sidebar.write(f"**Predictions:** {len(st.session_state.prediction_history)}")

if use_privacy:
    st.sidebar.markdown('<span class="info-badge">🔒 Privacy ON</span>', unsafe_allow_html=True)
if use_carbon_tracking:
    st.sidebar.markdown('<span class="info-badge">🌱 Carbon ON</span>', unsafe_allow_html=True)
if use_explanations:
    st.sidebar.markdown('<span class="info-badge">🔍 SHAP ON</span>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🧠 Responsible AI Framework | 
    CarbonTracker 🌱 | SHAP 🔍 | Flower 🌸 | Privacy 🔒
</div>
""", unsafe_allow_html=True)
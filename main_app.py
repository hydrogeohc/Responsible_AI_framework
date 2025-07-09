"""
Enhanced Streamlit Application for the Responsible AI Framework.
Complete interface for stress detection with carbon tracking, privacy preservation, 
federated learning, and model interpretability with street light system integration.
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
try:
    from src.stress_detection.core.framework import ResponsibleAIFramework
    from src.stress_detection.data.data_utils import get_sample_data_for_demo
    from src.stress_detection.data.time_series_utils import get_sample_time_series_for_demo
    from src.street_light import StreetLightResponsibleAI, StreetLightDataProcessor, create_sample_district_data
    from config import get_config
    STREET_LIGHT_AVAILABLE = True
except ImportError as e:
    st.error(f"Import error: {e}")
    STREET_LIGHT_AVAILABLE = False

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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .success-badge {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .info-badge {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .carbon-badge {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .system-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'framework' not in st.session_state:
    st.session_state.framework = None
if 'street_light_framework' not in st.session_state:
    st.session_state.street_light_framework = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_system' not in st.session_state:
    st.session_state.current_system = "Stress Detection"

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Responsible AI Framework</h1>
    <p>Comprehensive AI solutions for healthcare and smart cities with carbon tracking, privacy preservation, and explainable AI</p>
</div>
""", unsafe_allow_html=True)

# System selection
st.sidebar.header("🎯 System Selection")
system_choice = st.sidebar.selectbox(
    "Choose AI System",
    ["Stress Detection", "Street Light IoT"] if STREET_LIGHT_AVAILABLE else ["Stress Detection"],
    help="Select the AI system to use"
)

st.session_state.current_system = system_choice

# System-specific configuration
if system_choice == "Stress Detection":
    st.sidebar.header("⚙️ Stress Detection Configuration")
    
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
            with st.spinner("Initializing stress detection framework..."):
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

elif system_choice == "Street Light IoT" and STREET_LIGHT_AVAILABLE:
    st.sidebar.header("⚙️ Street Light Configuration")
    
    # Environment selection
    environment = st.sidebar.selectbox(
        "Environment",
        ["dev", "prod", "test"],
        help="Choose the configuration environment"
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["performance", "carbon", "led"],
        help="Choose the street light model type"
    )
    
    # Feature toggles
    use_privacy = st.sidebar.checkbox("Enable Privacy Protection", value=True)
    use_carbon_tracking = st.sidebar.checkbox("Enable Carbon Tracking", value=True)
    use_explanations = st.sidebar.checkbox("Enable SHAP Explanations", value=True)
    
    # Initialize street light framework
    if st.session_state.street_light_framework is None:
        try:
            with st.spinner("Initializing street light framework..."):
                config = get_config(environment)
                st.session_state.street_light_framework = StreetLightResponsibleAI(config=config)
                st.session_state.training_complete = False
            st.sidebar.success("✅ Street Light Framework ready!")
        except Exception as e:
            st.sidebar.error(f"❌ Initialization failed: {e}")
            st.stop()
    
    framework = st.session_state.street_light_framework

# Main content based on system choice
if system_choice == "Stress Detection":
    # Stress Detection Interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Training", 
        "🔮 Prediction", 
        "📊 Analytics", 
        "🌐 Federated Learning",
        "⚙️ System Status"
    ])
    
    # Tab 1: Training
    with tab1:
        st.header("🧠 Stress Detection Model Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Training Configuration")
            
            # Training parameters
            epochs = st.slider("Training Epochs", 5, 20, 10)
            
            # Data source
            data_source = st.selectbox(
                "Data Source",
                ["Synthetic Demo Data", "Time Series Data"]
            )
            
            # Start training
            if st.button("🚀 Start Training", type="primary"):
                try:
                    with st.spinner("Training with carbon tracking..."):
                        if data_source == "Synthetic Demo Data":
                            features, labels = get_sample_data_for_demo()
                        else:
                            features, labels = get_sample_time_series_for_demo()
                        
                        training_result = framework.train_with_carbon_tracking(
                            features, labels, 
                            epochs=epochs, 
                            use_privacy=use_privacy
                        )
                        
                        st.session_state.training_complete = True
                        st.success("✅ Training completed successfully!")
                        
                        # Display results
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Accuracy", f"{training_result['accuracy']:.1%}")
                        with col_b:
                            st.metric("CO₂ Emissions", f"{training_result['carbon_emissions']['co2_kg']:.6f} kg")
                        with col_c:
                            st.metric("Energy Used", f"{training_result['carbon_emissions']['energy_kwh']:.6f} kWh")
                        with col_d:
                            st.metric("Privacy", "Protected" if training_result['privacy_used'] else "Standard")
                        
                        # Training progress visualization
                        if hasattr(training_result, 'training_losses'):
                            fig = px.line(
                                x=range(len(training_result['training_losses'])),
                                y=training_result['training_losses'],
                                title="Training Loss Over Time"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
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
                <strong>📊 Framework Status</strong><br>
                <b>Model:</b> {status['model_type']}<br>
                <b>Training Runs:</b> {status['training_history']}<br>
                <b>Explanations:</b> {status['explanations_generated']}<br>
                <b>Privacy ε:</b> {status['privacy_epsilon']:.2f}<br>
                <b>SHAP Ready:</b> {status['shap_explainer_initialized']}
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 2: Prediction
    with tab2:
        st.header("🔮 Stress Level Prediction")
        
        if not st.session_state.training_complete:
            st.warning("⚠️ Please train the model first")
        else:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("Patient Input")
                
                # Input method
                input_method = st.radio(
                    "Input Method",
                    ["Manual Input", "Predefined Scenarios", "Upload Data"]
                )
                
                if input_method == "Manual Input":
                    # Manual input fields
                    col_a, col_b = st.columns(2)
                    with col_a:
                        age = st.slider("Age", 18, 80, 35)
                        height = st.slider("Height (cm)", 140, 220, 170)
                    with col_b:
                        weight = st.slider("Weight (kg)", 40, 150, 70)
                        activity = st.selectbox("Physical Activity Level", ["None", "Light", "Moderate", "High"])
                    
                    activity_val = {"None": 0, "Light": 1, "Moderate": 2, "High": 3}[activity]
                    input_data = np.array([[age, height, weight, activity_val]])
                    
                elif input_method == "Predefined Scenarios":
                    # Enhanced predefined scenarios
                    scenarios = {
                        "Healthy Young Adult": [25, 175, 70, 2],
                        "Sedentary Office Worker": [40, 170, 85, 0],
                        "Active Senior": [65, 165, 75, 1],
                        "Stressed Professional": [35, 172, 90, 0],
                        "Fitness Enthusiast": [30, 180, 75, 3],
                        "Healthcare Worker": [45, 165, 68, 1]
                    }
                    
                    selected_scenario = st.selectbox("Select Scenario", list(scenarios.keys()))
                    input_data = np.array([scenarios[selected_scenario]])
                    
                    # Display scenario details
                    st.info(f"**Profile:** Age: {input_data[0][0]}, Height: {input_data[0][1]}cm, Weight: {input_data[0][2]}kg, Activity: {input_data[0][3]}")
                
                else:  # Upload Data
                    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
                    if uploaded_file is not None:
                        df = pd.read_csv(uploaded_file)
                        st.write("Preview of uploaded data:")
                        st.dataframe(df.head())
                        
                        if st.button("Process All Rows"):
                            input_data = df.values
                        else:
                            input_data = df.values[:1]  # Use first row
                    else:
                        input_data = np.array([[35, 170, 70, 1]])  # Default
                
                # Prediction button
                if st.button("🔮 Predict Stress Level", type="primary"):
                    try:
                        with st.spinner("Generating prediction with explanations..."):
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
                            st.success(f"**🎯 Predicted Stress Level: {prediction_result['predicted_stress']}**")
                            st.write(f"**📊 Confidence: {prediction_result['confidence']:.1%}**")
                            
                            # Probability distribution
                            probs = prediction_result['probabilities']
                            prob_df = pd.DataFrame({
                                'Stress Level': ['Low', 'Medium', 'High'],
                                'Probability': [probs['low'], probs['medium'], probs['high']]
                            })
                            
                            fig = px.bar(
                                prob_df, 
                                x='Stress Level', 
                                y='Probability',
                                title="🎯 Stress Level Probability Distribution",
                                color='Probability',
                                color_continuous_scale='RdYlBu_r'
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # SHAP explanation
                            if use_explanations and 'explanation' in prediction_result:
                                explanation = prediction_result['explanation']
                                if 'error' not in explanation and 'feature_importance' in explanation:
                                    st.subheader("🔍 SHAP Feature Importance Analysis")
                                    
                                    importance_df = pd.DataFrame(explanation['feature_importance'])
                                    
                                    fig = px.bar(
                                        importance_df, 
                                        x='importance', 
                                        y='feature',
                                        orientation='h',
                                        title="🔍 Feature Impact on Prediction",
                                        color='shap_value',
                                        color_continuous_scale='RdBu_r'
                                    )
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Feature explanations
                                    st.write("**🧠 Feature Impact Analysis:**")
                                    for feat in importance_df.head(4).itertuples():
                                        impact_icon = "📈" if feat.impact == 'positive' else "📉"
                                        st.write(f"{impact_icon} **{feat.feature}**: {feat.shap_value:.3f} ({'increases' if feat.impact == 'positive' else 'decreases'} stress)")
                                else:
                                    st.info("Using gradient-based explanation fallback")
                            
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                        logger.error(f"Prediction error: {e}")
            
            with col2:
                st.subheader("📝 Recent Predictions")
                
                if st.session_state.prediction_history:
                    recent_predictions = st.session_state.prediction_history[-5:]
                    
                    for i, pred in enumerate(reversed(recent_predictions)):
                        with st.expander(f"Prediction {len(recent_predictions) - i}"):
                            result = pred['result']
                            st.write(f"**Result:** {result['predicted_stress']}")
                            st.write(f"**Confidence:** {result['confidence']:.1%}")
                            st.write(f"**Time:** {pred['timestamp'][:19]}")
                            
                            # Mini probability chart
                            probs = result['probabilities']
                            mini_df = pd.DataFrame({
                                'Level': ['Low', 'Med', 'High'],
                                'Prob': [probs['low'], probs['medium'], probs['high']]
                            })
                            fig = px.bar(mini_df, x='Level', y='Prob', height=200)
                            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No predictions yet")
    
    # Tab 3: Analytics
    with tab3:
        st.header("📊 Analytics Dashboard")
        
        if not st.session_state.prediction_history:
            st.info("📈 No analytics data available. Make predictions to see analytics.")
        else:
            # Process prediction history
            predictions_df = pd.DataFrame([
                {
                    'timestamp': pred['timestamp'],
                    'predicted_stress': pred['result']['predicted_stress'],
                    'confidence': pred['result']['confidence'],
                    'age': pred['input'][0][0],
                    'height': pred['input'][0][1],
                    'weight': pred['input'][0][2],
                    'activity': pred['input'][0][3]
                }
                for pred in st.session_state.prediction_history
            ])
            
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            
            # Key metrics
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
            
            # Advanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Stress distribution pie chart
                stress_counts = predictions_df['predicted_stress'].value_counts()
                fig = px.pie(
                    values=stress_counts.values, 
                    names=stress_counts.index,
                    title="📊 Stress Level Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Age vs Stress scatter plot
                fig = px.scatter(
                    predictions_df, 
                    x='age', 
                    y='confidence',
                    color='predicted_stress',
                    title="🎯 Age vs Prediction Confidence"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence over time
                fig = px.line(
                    predictions_df, 
                    x='timestamp', 
                    y='confidence',
                    title="📈 Prediction Confidence Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # BMI vs Stress analysis
                predictions_df['bmi'] = predictions_df['weight'] / (predictions_df['height'] / 100) ** 2
                fig = px.box(
                    predictions_df, 
                    x='predicted_stress', 
                    y='bmi',
                    title="📊 BMI Distribution by Stress Level"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Carbon footprint analysis
            if framework.training_history:
                st.subheader("🌱 Carbon Footprint Analysis")
                
                carbon_data = []
                for i, training in enumerate(framework.training_history):
                    carbon_data.append({
                        'run': i + 1,
                        'co2_kg': training['carbon_emissions']['co2_kg'],
                        'energy_kwh': training['carbon_emissions']['energy_kwh'],
                        'accuracy': training['accuracy']
                    })
                
                carbon_df = pd.DataFrame(carbon_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        carbon_df, 
                        x='run', 
                        y='co2_kg',
                        title="🌍 CO₂ Emissions by Training Run"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        carbon_df, 
                        x='accuracy', 
                        y='co2_kg',
                        size='energy_kwh',
                        title="⚖️ Accuracy vs Carbon Footprint"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Federated Learning
    with tab4:
        st.header("🌐 Federated Learning Simulation")
        
        st.markdown("""
        <div class="feature-highlight">
            <h4>🤝 Multi-Institution Collaboration</h4>
            <p>Simulate federated learning across multiple healthcare institutions while preserving patient privacy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Federated Learning Configuration")
            
            num_clients = st.slider("Number of Healthcare Institutions", 2, 8, 3)
            fed_rounds = st.slider("Federated Learning Rounds", 1, 10, 3)
            samples_per_client = st.slider("Samples per Institution", 20, 200, 50)
            
            # Institution types
            institution_types = st.multiselect(
                "Institution Types",
                ["Hospital", "Clinic", "Research Center", "Wellness Center"],
                default=["Hospital", "Clinic", "Research Center"]
            )
            
            if st.button("🚀 Start Federated Learning", type="primary"):
                try:
                    with st.spinner("Running federated learning simulation..."):
                        # Generate client data
                        client_data = []
                        for i in range(num_clients):
                            features, labels = get_sample_data_for_demo()
                            # Add some variation per client
                            features = features + np.random.normal(0, 0.1, features.shape)
                            client_data.append((features[:samples_per_client], labels[:samples_per_client]))
                        
                        fed_result = framework.federated_learning(client_data, rounds=fed_rounds)
                        
                        if fed_result['federated_training']['success']:
                            st.success("✅ Federated learning completed successfully!")
                            
                            # Display results
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Institutions", fed_result['federated_training']['clients'])
                            with col_b:
                                st.metric("Rounds", fed_result['federated_training']['rounds'])
                            with col_c:
                                st.metric("Privacy", "Protected" if fed_result['privacy_protected'] else "Standard")
                            
                            # Show federated learning progress
                            if 'results' in fed_result['federated_training']:
                                results_df = pd.DataFrame(fed_result['federated_training']['results'])
                                
                                fig = px.line(
                                    results_df, 
                                    x='round', 
                                    y=['loss', 'accuracy'],
                                    title="🌐 Federated Learning Progress"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("❌ Federated learning failed")
                            
                except Exception as e:
                    st.error(f"❌ Federated learning failed: {e}")
        
        with col2:
            st.subheader("Federated Learning Benefits")
            
            st.markdown("""
            <div class="metric-card">
                <h4>🔒 Privacy Preservation</h4>
                <p>Patient data never leaves the institution</p>
            </div>
            
            <div class="metric-card">
                <h4>🤝 Collaborative Learning</h4>
                <p>Share knowledge without sharing data</p>
            </div>
            
            <div class="metric-card">
                <h4>📈 Better Models</h4>
                <p>Improved accuracy through collaboration</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 5: System Status
    with tab5:
        st.header("⚙️ System Status")
        
        status = framework.get_status()
        
        # System overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="system-card">
                <h4>🧠 Model Status</h4>
                <b>Type:</b> {status['model_type']}<br>
                <b>Trained:</b> {status['model_trained']}<br>
                <b>Training Runs:</b> {status['training_history']}<br>
                <b>Explanations:</b> {status['explanations_generated']}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="system-card">
                <h4>🔒 Privacy Status</h4>
                <b>Epsilon:</b> {status['privacy_epsilon']:.2f}<br>
                <b>Protection:</b> {'On' if use_privacy else 'Off'}<br>
                <b>Mechanism:</b> Laplace DP<br>
                <b>K-Anonymity:</b> Available
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="system-card">
                <h4>🌱 Carbon Tracking</h4>
                <b>Active:</b> {status['carbon_tracking_active']}<br>
                <b>Monitoring:</b> {'On' if use_carbon_tracking else 'Off'}<br>
                <b>Component:</b> CarbonTracker<br>
                <b>Logs:</b> Available
            </div>
            """, unsafe_allow_html=True)
        
        # Component status
        st.subheader("Component Status")
        
        components = [
            ("🔍 SHAP Explainer", status['shap_explainer_initialized']),
            ("🌸 Flower Federated Learning", status['flower_fl_initialized']),
            ("🌱 Carbon Tracker", status['carbon_tracking_active']),
            ("🔒 Privacy Layer", True)
        ]
        
        for name, active in components:
            badge_class = "success-badge" if active else "warning-badge"
            status_text = "✅ Active" if active else "⚠️ Inactive"
            st.markdown(f'{name}: <span class="{badge_class}">{status_text}</span>', unsafe_allow_html=True)
        
        # Privacy report
        st.subheader("🔒 Privacy Report")
        privacy_report = framework.privacy_layer.get_privacy_report()
        st.json(privacy_report)
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        if framework.training_history:
            latest_training = framework.training_history[-1]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Latest Accuracy", f"{latest_training['accuracy']:.1%}")
            with col2:
                st.metric("Latest CO₂", f"{latest_training['carbon_emissions']['co2_kg']:.6f} kg")
            with col3:
                st.metric("Latest Energy", f"{latest_training['carbon_emissions']['energy_kwh']:.6f} kWh")

elif system_choice == "Street Light IoT" and STREET_LIGHT_AVAILABLE:
    # Street Light Interface
    st.header("🏙️ Street Light IoT Management")
    
    tab1, tab2, tab3 = st.tabs([
        "🎯 Training & Prediction",
        "📊 Analytics",
        "⚙️ System Status"
    ])
    
    with tab1:
        st.subheader("Street Light Model Training")
        
        # Create sample street light data
        data_processor = StreetLightDataProcessor()
        sample_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100, freq='M'),
            'Cumulative # of streetlights converted to LED': np.random.randint(1000, 200000, 100),
            '% outages repaired within 10 business days': np.random.randint(75, 99, 100)
        })
        
        if st.button("🚀 Train Street Light Model"):
            try:
                with st.spinner("Training street light model..."):
                    result = framework.train_performance_model(sample_data, epochs=5)
                    
                    if 'error' not in result:
                        st.success("✅ Street light model trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{result['accuracy']:.1%}")
                        with col2:
                            st.metric("CO₂ Emissions", f"{result['carbon_emissions']['co2_kg']:.6f} kg")
                        with col3:
                            st.metric("Energy Used", f"{result['carbon_emissions']['energy_kwh']:.6f} kWh")
                    else:
                        st.error(f"❌ Training failed: {result['error']}")
            
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
        
        # Prediction section
        st.subheader("Street Light Prediction")
        
        if st.button("🔮 Predict Street Light Performance"):
            try:
                prediction = framework.predict_with_explanation(sample_data.head(1))
                
                if 'error' not in prediction:
                    st.success(f"**Predicted Status: {prediction['predicted_status']}**")
                    st.write(f"**Confidence: {prediction['confidence']:.1%}**")
                    st.write(f"**Energy Consumption: {prediction['energy_consumption_kwh']:.2f} kWh**")
                    st.write(f"**Carbon Footprint: {prediction['carbon_footprint_kg']:.4f} kg CO2**")
                else:
                    st.error(f"❌ Prediction failed: {prediction['error']}")
            
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
    
    with tab2:
        st.subheader("Street Light Analytics")
        
        # Display sample analytics
        st.info("Street light analytics would show LED conversion progress, energy savings, and maintenance predictions.")
        
        # Sample chart
        sample_chart_data = pd.DataFrame({
            'Month': pd.date_range('2020-01-01', periods=12, freq='M'),
            'LED_Conversions': np.random.randint(1000, 5000, 12),
            'Energy_Savings': np.random.randint(500, 2000, 12)
        })
        
        fig = px.line(sample_chart_data, x='Month', y=['LED_Conversions', 'Energy_Savings'],
                     title="LED Conversion Progress")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Street Light System Status")
        
        status = framework.get_framework_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="system-card">
                <h4>🏙️ Street Light Models</h4>
                <b>Performance:</b> {status['models']['performance_model_trained']}<br>
                <b>Carbon:</b> {status['models']['carbon_model_trained']}<br>
                <b>LED:</b> {status['models']['led_model_trained']}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="system-card">
                <h4>🌱 Environmental Impact</h4>
                <b>Carbon Tracking:</b> {status['carbon_tracking']['carbon_tracker_active']}<br>
                <b>Privacy:</b> {status['privacy']['epsilon']}<br>
                <b>Federated Learning:</b> {status['federated_learning']['flower_initialized']}
            </div>
            """, unsafe_allow_html=True)

# Sidebar status
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Status")

if system_choice == "Stress Detection":
    status = framework.get_status()
    st.sidebar.write(f"**Model:** {status['model_type']}")
    st.sidebar.write(f"**Trained:** {'Yes' if status['model_trained'] else 'No'}")
    st.sidebar.write(f"**Predictions:** {len(st.session_state.prediction_history)}")
    
    if use_privacy:
        st.sidebar.markdown('<span class="info-badge">🔒 Privacy ON</span>', unsafe_allow_html=True)
    if use_carbon_tracking:
        st.sidebar.markdown('<span class="carbon-badge">🌱 Carbon ON</span>', unsafe_allow_html=True)
    if use_explanations:
        st.sidebar.markdown('<span class="info-badge">🔍 SHAP ON</span>', unsafe_allow_html=True)

elif system_choice == "Street Light IoT" and STREET_LIGHT_AVAILABLE:
    st.sidebar.write(f"**System:** Street Light IoT")
    st.sidebar.write(f"**Environment:** {environment}")
    st.sidebar.write(f"**Model Type:** {model_type}")
    
    st.sidebar.markdown('<span class="info-badge">🏙️ Smart City</span>', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="carbon-badge">🌱 Sustainable</span>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 1rem;">
    🧠 Responsible AI Framework | Current System: <strong>{system_choice}</strong> |
    CarbonTracker 🌱 | SHAP 🔍 | Flower 🌸 | Privacy 🔒 | 
    Built with ❤️ for Healthcare & Smart Cities
</div>
""", unsafe_allow_html=True)
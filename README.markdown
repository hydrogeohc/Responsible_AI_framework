# Federated Trustworthy AI Framework

This repository provides a modular framework for building, monitoring, and demonstrating responsible federated AI systems. It integrates federated learning (Flower), privacy-preserving computation (PySyft, Differential Privacy), model interpretability (SHAP), carbon emission tracking (Carbontracker), and multi-layered security—all accessible via an interactive Streamlit dashboard.

## 📁 Project File Structure

```
Federated_trustworthy_ai/
│
├── app.py                    # Streamlit dashboard frontend
├── models.py                 # Model definitions (e.g., PyTorch models)
├── client.py                 # Federated learning client logic (Flower)
├── server.py                 # Federated learning server logic (Flower)
├── carbon_emission_layer.py  # Carbon emission calculation utilities (Carbontracker)
├── interpretability_utils.py # Model interpretability utilities (SHAP)
├── privacy_utils.py          # Privacy-preserving computation utilities (PySyft, DP)
├── security_layer.py         # Security utilities (encryption, access control)
└── requirements.txt          # Python dependencies

```

## 🚀 Quick Start

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Streamlit dashboard:**
    ```bash
    streamlit run app.py
    ```

3. **Start the federated learning server and clients (in separate terminals):**
    ```bash
    python server.py
    python client.py
    ```

## 🛠️ Features

- **Federated Learning:** Orchestrate decentralized AI training with Flower.
- **Privacy-Preserving Computation:** Secure data sharing with PySyft, differential privacy, and homomorphic encryption.
- **Model Interpretability:** Explain model predictions using SHAP.
- **Carbon Emission Calculation:** Track and report the environmental impact of computation with Carbontracker.
- **Security:** Multi-layered defense, including encryption, anonymization, access control, and secure aggregation.

## 📚 Customization

- Replace or extend models in `models.py` for the healthcare, cities use case.
- Connect real backend logic to the Streamlit UI by implementing functions in the respective utility modules.
- Adapt privacy and security layers to the use case data governance requirements.

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

## Current contributors
- Ying Jung Chen
- Fan-Ying Lin

## 📝 License

MIT License

**This framework is designed for research, prototyping, and demonstration of responsible, privacy-aware, and sustainable federated AI systems.**
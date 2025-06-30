import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_model(model, data):
    # SHAP expects numpy arrays for most explainers; DeepExplainer for PyTorch
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)
    return shap_values

def plot_shap(shap_values, data):
    # Convert to numpy for visualization if needed
    shap.image_plot(shap_values, data.numpy())
    plt.show()

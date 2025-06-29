import shap
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import numpy as np

# Simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))

def main():
    # Load test data (MNIST)
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = Net()
    model.eval()  # Set to evaluation mode
    
    # Get explanation data
    images, _ = next(iter(testloader))
    X_explain = images[:10]  # First 10 images
    
    # SHAP explainer
    explainer = shap.DeepExplainer(model, torch.zeros(1, 1, 28, 28))
    shap_values = explainer.shap_values(X_explain)
    
    # Convert to numpy for visualization
    shap_np = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
    X_explain_np = np.swapaxes(np.swapaxes(X_explain.numpy(), 2, 3), 1, -1)
    
    # Visualization
    shap.image_plot(shap_np, -X_explain_np)
    print("SHAP analysis completed. Check plot for results.")

if __name__ == "__main__":
    main()
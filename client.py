import flwr as fl
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from codecarbon import EmissionsTracker
import shap
import syft as sy
import numpy as np

# --- PySyft Setup for Secure Computation ---
hook = sy.TorchHook(torch)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# --- Model Definition ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))

# --- Data Loading (MNIST as Example) ---
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# --- Model, Optimizer, Loss ---
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# --- CodeCarbon Emissions Tracker ---
tracker = EmissionsTracker()
tracker.start()

# --- Flower Client Definition ---
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]
    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        return self.get_parameters(config={}), len(trainloader.dataset), {}
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
    tracker.stop()
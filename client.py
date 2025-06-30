import flwr as fl
import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim, nn
from models import Net
from privacy_utils import get_virtual_workers, share_tensor
from interpretability_utils import explain_model, plot_shap

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Privacy: Share a tensor example (not the model weights, just demo)
alice, bob = get_virtual_workers()
x = torch.tensor([1, 2, 3, 4, 5])
x_shared = share_tensor(x, [alice, bob])

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
        # Interpretability: Explain a batch
        images, _ = next(iter(testloader))
        shap_values = explain_model(model, images[:10])
        plot_shap(shap_values, images[:10])
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())

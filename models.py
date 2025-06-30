import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))
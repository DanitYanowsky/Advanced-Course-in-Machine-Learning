import torch.nn as nn
class LinearProbing(nn.Module):
    def __init__(self, d):
        super(LinearProbing, self).__init__()
        self.fc = nn.Linear(d, 10)


    def forward(self, x):
        return self.fc(x)
import torch
import torch.nn as nn

class Generator(nn.Module):
    """3-layer fcn
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_features),
            nn.Sigmoid()
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.view(-1, 1, 28, 28) #(batch, channel, w, h)
        return x

class Discriminator(nn.Module):
    """3-layer fcn with LeakyReLU. 
    """
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28*28) # since our network is fcn
        x = self.net(x)
        x = x.squeeze(dim=1) # vector
        return x
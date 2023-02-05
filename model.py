import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net_g = nn.Sequential(
                    nn.Linear(2, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2),
                    nn.ReLU()
                    ).to(device)

    def forward(self, x):
        return self.net_g(x)


class Discrimator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net_d = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, x):
        return self.net_d(x)



class System(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.generator =  Generator(device)
        self.discriminator = Discrimator(device)

    def forward(self, x):
        return self.discriminator(self.generator(x))

    def generate_samples(self, x):
        return self.generator(x)

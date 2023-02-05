import torch
from model import System
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from train import Trainer
torch.manual_seed(42)

torch.set_default_dtype(torch.float64)

if __name__=='__main__':

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print("Using device {}".format(device))

    model = System(device=device)

    trainer = Trainer(model)
    trainer.train()
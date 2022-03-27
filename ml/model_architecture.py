import torch
from torch import nn
from torch.nn import Module
from torch.cuda import init
from torch.utils.data import Dataset


class ClassificationModel(Module):
    def __init__(self, num_features):
        super().__init__()
        fc1 = nn.Linear(num_features, 128)
        fc2 = nn.Linear(128, 64)
        fc3 = nn.Linear()
        
        
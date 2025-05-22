
import torch
import torch.nn as nn

# linear model for full transparency
class LearnableSimilarityModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

import torch
import torch.nn as nn
import numpy as np

class MultiplyByScalarLayer(nn.Module):
    # A simple layer to multiply all entries by a constant scalar value. Needed since action inputs are not normalized in
    # many environments and tanh is then critical, unlike in D4RL where actions are in [-1, 1].
    # scalar value should be absolute max possible action value.

    def __init__(self, scalar):
        super(MultiplyByScalarLayer, self).__init__()
        self.scalar = scalar

    def forward(self, tensors):
        result = torch.clone(tensors)
        for ind in range(result.shape[0]):
            result[ind] = torch.mul(result[ind], self.scalar)
        return result
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNet(nn.Module):
    def __init__(self, input_size, action_size, hidden_layers):
        super().__init__()
        
        layers = []
        
        input = input_size
        for i, units in enumerate(hidden_layers):
            output = units
            layers.append(nn.Linear(input, output))
            input = output
        
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(input, action_size)
        
    def forward(self, obs):
        x = obs
        for layer in self.layers:
            x = F.relu(layer(x))
            
        x = self.output(x)
        
        return x
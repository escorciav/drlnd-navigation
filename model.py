import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    "Actor (Policy) Model"

    def __init__(self, state_size, action_size, seed, hidden_units=64,
                 num_layers=1):
        """Initialize parameters and build model

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_units (int): Number of neurons in linear layer
            num_layers (int): Depth of the network
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        # Input layer
        backbone = [nn.Linear(state_size, self.hidden_units),
                    nn.ReLU(inplace=True)]
        # More capacity
        for i in range(self.num_layers):
            backbone += [nn.Linear(self.hidden_units, self.hidden_units),
                            nn.ReLU(inplace=True)]
        self.brain = nn.Sequential(
            *backbone,
            # Output layer
            nn.Linear(self.hidden_units, action_size)
        )

    def forward(self, state):
        "Build a network that maps state -> action values"
        x = self.brain(state)
        return x

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from .initialize import *


class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, state_space, activation):
        super(ValueNetworkBase, self).__init__()
        if isinstance(state_space, int): # pass in state_dim rather than state_space
            self._state_dim = state_space
        else:
            self._state_space = state_space
            self._state_shape = state_space.shape
            if len(self._state_shape) == 1:
                self._state_dim = self._state_shape[0]
            else:  # high-dim state
                pass  

        self.activation = activation

    def forward(self):
        pass

class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation ):
        super().__init__( state_space, activation)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = self._action_shape[0]
     

class QNetwork(QNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, output_dim=1, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, output_dim)
        # weights initialization
        # self.linear4.apply(linear_weights_init)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.linear4.weight)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x        

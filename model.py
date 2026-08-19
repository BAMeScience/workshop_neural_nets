# import all necessary packages
import numpy as np
import pandas as pd

# import relevant pytorch functions and packages
import torch
from torch import nn

################################################
################## The Model ###################
################################################

# We create a simple fully connected network.
# The input is our image, the output is a vector with 10 entries, 
# each entry has a value corresponding to a digit.
class MNISTClassifier_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Flatten(),               # flattens 28x28 vector into 1D with [B, 784]
            nn.Linear(28 * 28, 128),    # fully connected layer from 784 to 128 
            nn.ReLU(),                  # ReLu() activation 
            nn.Linear(128, 10),         # fully connected layer from 128 to 10 -> our 10 digits
        )

    def forward(self, x):
        
        network_result = self.network(x) # call the defined network 
        
        return network_result
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class Action_Predicter_Dense(nn.Module):
    def __init__(self):
        super(Action_Predicter_Dense, self).__init__()

        self.d1 = nn.Linear(64, 128)
        self.d2 = nn.Linear(128, 64)
        self.d3 = nn.Linear(64, 16)
        self.d4 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.d1(x)
        #print('d1')
        x = F.relu(x)
        #print('relu')
        x = self.d2(x)
        #print('d1')
        x = F.relu(x)
        #print('relu')
        x = self.d3(x)
        #print('d1')
        x = F.relu(x)
        #print('relu')
        x = self.d4(x)
        #print('d1')
        out = F.tanh(x)

        return out

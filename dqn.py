import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class DQNNet(nn.Module):
    def __init__(self,state_shape,action_shape,lr=1e-3):

        super(DQNNet,self).__init__()
        self.dense1 = nn.Linear(in_features=state_shape,out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=64)
        self.dense3 = nn.Linear(in_features=64,out_featurs=action_shape)
        
        self.optimizer = optim.Adam(self.parameters(),lr = lr)

        def forward(self,x):
            x = F.relu(self.dense1(x))
            x = F.relu(self.dense2(x))
            x = self.dense3(x)
            return x

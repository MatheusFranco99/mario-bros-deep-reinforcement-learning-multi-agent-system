import random
import re
import torch
import numpy as np

class ReplayMemory:
    def __init__(self,capacity):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.idx = 0
    
    def store(self,states,actions,next_states,rewards,done):
        if(len(self.states) < self.capacity):
            self.states.append(states)
            self.actions.append(actions)
            self.next_states.append(next_states)
            self.rewards.append(rewards)
            self.dones.append(done)
        else:
            self.states[self.idx] = states
            self.actions[self.idx] = actions
            self.next_states[self.idx] = next_states
            self.rewards[self.idx] = rewards
            self.dones[self.idx] = done
        
        self.idx = (self.idx+1) % self.capacity
    
    def sample(self,batchsize,device):
        indices_to_sample = random.sample(range(len(self.states)),k = batchsize)

        states = torch.from_numpy(np.array(self.states)[indices_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.actions)[indices_to_sample]).to(device)
        next_states = torch.from_numpy(np.array(self.next_states)[indices_to_sample]).float().to(device)
        rewards = torch.from_numpy(np.array(self.rewards)[indices_to_sample]).float().to(device)
        dones = torch.from_numpy(np.array(self.dones)[indices_to_sample]).to(device)
        
        return states,actions,next_states,rewards,dones
    
    def __len__(self):
        return len(self.states)
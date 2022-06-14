
import numpy as np
import random
from tensorflow.keras.models import load_model

from ReplayMemory import ReplayMemory
from DQNNet import DQNNet

import torch

import copy

class Agent:

    def __init__(self,
                n_actions,
                observation_dim,
                fname = "agent",
                gamma = 0.95,
                learning_rate = 0.01,
                epsilon_max=1.0, epsilon_min = 0.01, epsilon_dec = 0.995):

        self.n_actions = n_actions
        self.fname = fname
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec

        self.Q = torch.nn.Sequential(
            torch.nn.Linear(observation_dim,24),
            torch.nn.ReLu(),
            torch.nn.Linear(24, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, self.n_actions)
        ).doulble()

        self.Q_ = copy.deepcopy(self.Q)

        self.optimizer = torch.optim.Adam(self.Q.parameters(),lr = learning_rate)

    
    def choose_action(self,state, deterministic = False):

        # exploration vs. exploitation
        if not deterministic:
            random_value = np.random.rand()
            if random_value < self.epsilon:
                return np.random.randint(self.n_actions)
            else:
                with torch.no_grad():
                    return np.argmax(self.Q(state))
        else:
            with torch.no_grad():
                return np.argmax(self.Q(state))

    def learn(self, state, action, next_state, reward, done):
        
            
    
    def 

    def __init__(self,height,widht,num_frames,n_actions,epsilon=1,batch_size=32,alpha=0.0005,gamma=0.996,epsilon_step=1/(1e6),epsilon_min=0.1,mem_size=1000000,fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.height = height
        self.width = widht
        self.num_frames = num_frames
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.model_file = fname
        self.memory = ReplayMemory(mem_size)
        
        self.dqnnet = DQNNet(height,widht,num_frames,n_actions)
        self.dqnnet.build()
        self.dqnnet.compile(alpha)

    def choose_action(self,state,deterministic = False):

        state = state[np.newaxis,:]

        if deterministic:
            actions = self.dqnnet.predict(state)
            action = np.argmax(actions)
        else:
            rand = np.random.random()
            if(rand < self.epsilon):
                action = np.random.choice(self.action_space)
            else:
                actions = self.dqnnet.predict(state)
                action = np.argmax(actions)
        return action
    
    def remember(self,state,action,next_state,reward,done):
        self.memory.store(self.memory.toElement(state,action,next_state,reward,done))
    
    def learn(self):
        if(len(self.memory) < self.batch_size):
            return
        # fill up the memory with random actions

        mem_sample = self.memory.sample(self.batch_size)
    
        state = self.memory.getField(mem_sample,name='state')
        action = self.memory.getField(mem_sample,name='action')
        next_state = self.memory.getField(mem_sample,name='next_state')
        reward = self.memory.getField(mem_sample,name='reward')
        terminal = self.memory.getField(mem_sample,name='terminal')

        action_indices = action
        # print('inside learn')

        q_eval = self.dqnnet.predict(state)
        q_next = self.dqnnet.predict(next_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype = np.int32)

        
        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1)*terminal
        
        _ = self.dqnnet.model.fit(state,q_target,verbose=0)
    
        if(self.epsilon > self.epsilon_min):
            self.epsilon -= self.epsilon_step

    def save_model(self):
        self.dqnnet.save(self.model_file)
    
    def load_model(self):
        self.dqnnet.load(self.model_file)
   
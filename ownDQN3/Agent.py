
from pickletools import optimize
import numpy as np
from tensorflow.keras.models import load_model

from ReplayMemory import ReplayMemory
from DQNNet import DQNNet

import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

import copy

class Agent:

    def __init__(self,
                n_actions,
                observation_dim,
                fname = "agent",
                memsize = 1000000,
                batchsize = 20,
                gamma = 0.95,
                learning_rate = 0.01,
                epsilon_max=1.0, epsilon_min = 0.01, epsilon_dec = 0.995):
        self.n_actions = n_actions
        self.observation_dim = observation_dim
        self.fname = fname
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.memsize = memsize
        self.batchsize = batchsize

        self.replayBuffer = ReplayBuffer(self.memsize)

        self.Q = Sequential()
        self.Q.add(Dense(self.observation_dim,activation='relu'))
        self.Q.add(Dense(24,activation='relu'))
        self.Q.add(Dense(24,activation='relu'))
        self.Q.add(Dense(self.n_actions,activation='relu'))
        self.Q.compile(optimizer = Adam(learning_rate=learning_rate),loss='mse')

        self.Q_ = copy.deepcopy(self.Q)

    def pickRandomAction(self):
        return np.random.randint(self.n_actions)

    def pickAction(self,state):
        actions = self.Q.predict(state)
        return np.argmax(actions)

    def store(self,state,action,next_state,reward,done):
        self.replayBuffer.store(state,action,next_state,reward,done)

    def learn(self):
        sample = self.replayBuffer(self.batchsize)
        y = 

    def updateQ(self,):
        self.Q_ = self.Q
        self.Q_ = copy.deepcopy(self.Q)

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
   
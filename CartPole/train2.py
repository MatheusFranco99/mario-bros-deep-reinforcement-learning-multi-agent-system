from re import I
import gym
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import random

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Input
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.models import load_model


class ReplayBuffer:

    def __init__(self,mem_size):
        self.idx = 0
        self.capacity = mem_size
        self.memory = []
    
    def store(self,state,action,next_state,reward,done):
        if(len(self.memory) < self.capacity):
            self.memory.append((state,action,next_state,reward,done))
        else:
            self.memory[self.idx] = (state,action,next_state,reward,done)
            self.idx = (self.idx + 1)%self.capacity
    
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:

    def __init__(self,observation_dim,n_actions,
                    mem_size = 3000,
                    gamma = 0.95,
                    learning_rate = 0.00025,
                    epsilon_max = 1.0, epsilon_min = 0.001, epsilon_dec = 0.999):
        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.mem_size = mem_size
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.replayBuffer = ReplayBuffer(self.mem_size)

        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec


        # self.Q = Sequential()
        # self.Q.add(Dense(24,input_shape=(self.observation_dim,),activation="relu"))
        # self.Q.add(Dense(24,activation="relu"))
        # self.Q.add(Dense(self.n_actions,activation="linear"))
        # self.Q.compile(loss='mse',optimizer = Adam(learning_rate = self.learning_rate))

        X_input = Input((self.observation_dim,))

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 512 nodes
        X = Dense(512, input_shape=(self.observation_dim,), activation="relu", kernel_initializer='he_uniform')(X_input)

        # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
        
        # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(self.n_actions, activation="linear", kernel_initializer='he_uniform')(X)

        model = Model(inputs = X_input, outputs = X)#, name='CartPole DQN model')
        model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        self.Q = model

    def choose_action(self,state,deterministic = False):


        if deterministic:
            q_values = self.Q.predict(state,verbose = 0)
            return np.argmax(q_values)
        else:
            if np.random.random() < self.epsilon:
                return random.randint(0,self.n_actions-1)
            else:
                q_values = self.Q.predict(state,verbose = 0)
                return np.argmax(q_values)

    def store(self,state,action,next_state,reward,done):
        self.replayBuffer.store(state,action,next_state,reward,done)

    
    def update_epsilon(self):
        if(self.epsilon > self.epsilon_min):
            self.epsilon = self.epsilon*self.epsilon_dec

    def learn(self,batch_size):
        if(len(self.replayBuffer) < batch_size or len(self.replayBuffer) < 1000):
            return
        
        sample = self.replayBuffer.sample(batch_size)


        state = np.zeros((batch_size, self.observation_dim))
        next_state = np.zeros((batch_size, self.observation_dim))
        action, reward, done = [], [], []
        action = []
        reward = []
        done = []

        for i in range(batch_size):
            state[i] = sample[i][0]
            action.append(sample[i][1])
            next_state[i] = sample[i][2]
            reward.append(sample[i][3])
            done.append(sample[i][4])

        q_eval = self.Q.predict(state,verbose = 0)
        q_eval_next = self.Q.predict(next_state, verbose = 0)

        for i in range(batch_size):
            if done[i]:
                q_eval[i][action[i]] = reward[i]
            else:
                q_eval[i][action[i]] = reward[i] + self.gamma * (np.amax(q_eval_next[i]))

        # Train the Neural Network with batches
        self.Q.fit(state, q_eval, batch_size=batch_size, verbose=0)
        self.update_epsilon()




def trainCartPole(n_episodes = 1000, batch_size = 64):
    env = gym.make("CartPole-v1")

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = DQNAgent(observation_space,action_space)

    scores = []

    for episode in range(n_episodes):
        state = env.reset()
        state = np.reshape(state,[1,len(state)])

        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(state,[1,len(next_state)])
            total_reward += reward
            

            if(done and total_reward != env._max_episode_steps -1):
                reward = -100
            
            agent.store(state,action,next_state,reward,done)

            state = next_state

            if done:
                scores += [total_reward]
                print(f"Epsiode {episode} eneded with score {total_reward}, e {agent.epsilon}")
                # if(total_reward == 500):
                #     return agent, scores
                break

            agent.learn(batch_size)
    
    return agent,scores



agent, scores = trainCartPole(n_episodes = 1000)

import pickle

f = open("mine","wb")
pickle.dump(agent,f)
f.close()

print(scores)
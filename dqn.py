
import random
from time import sleep
import numpy as np

from pettingzoo.atari import mario_bros_v3


class ReplayMemory:

    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.memory = []
        self.idx = 0
    
    def toElement(self,state,action,next_state,reward,terminal):
        return {'state':state, 'action':action, 'next_state':next_state,'reward':reward,'terminal': 1-int(terminal)}

    def fromElement(self,elm):
        return elm['state'],elm['action'],elm['next_state'],elm['reward'],elm['terminal']
    
    def store(self,elm):
        if(len(self.memory) < self.capacity):
            self.memory += [elm]
        else:
            self.memory[self.idx] = elm
            self.idx = (self.idx + 1) % self.capacity
    
    def sample(self,batchsize):
        indices_to_sample = random.sample(range(len(self.memory)),k = batchsize)

        return np.array(self.memory)[indices_to_sample]

    def __len__(self):
        return len(self.memory)
    
    def getField(self, memory, name = "state"):
        ans = []
        for i in range(len(memory)):    
            ans += [memory[i][name]]
        return np.array(ans)
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

class DQNNet:

    def __init__(self,height,width,num_frames,num_actions):
    
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_actions = num_actions

    def build(self):
        model = Sequential()
        model.add(Convolution2D(16,(8,8),strides = (4,4), activation = 'relu',input_shape=(self.num_frames,self.height,self.width,1)))
        model.add(Convolution2D(32,(8,8),strides = (4,4), activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(self.num_actions,activation='linear'))
        self.model = model
        return self.model

    def compile(self,lr):
        self.model.compile(optimizer = Adam(lr=lr),loss='mse')
        return self.model
    
    def predict(self,state):
        actions = self.model.predict(state)
        return actions
    


class Agent:
    def __init__(self,height,widht,num_frames,n_actions,epsilon,batch_size,alpha=0.0005,gamma=0.996,epsilon_step=0.00001,epsilon_min=0.01,mem_size=1000000,fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayMemory(mem_size)
        
        self.dqnnet = DQNNet(height,widht,num_frames,n_actions)
        self.dqnnet.build()
        self.dqnnet.compile(alpha)

    def choose_action(self,state):

        # state = state[np.newaxis,:]

        rand = np.random.random()
        if(rand < self.epsilon):
            action = np.random.choice(self.action_space)
        else:
            action = self.dqnnet.predict(state)
        
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
        print('inside learn')

        q_eval = self.dqnnet.predict(state)
        q_next = self.dqnnet.predict(next_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype = np.int32)

        
        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1)*terminal
        
        _ = self.dqnnet.model.fit(state,q_target,verbose=0)
    
        if(self.epsilon > self.epsilon_min):
            self.epsilon -= self.epsilon_step

    def save_model(self):
        self.q_eval.save(self.model_file)
    
    def load_model(self):
        self.q_eval = load_model(self.model_file)
    

def normal_step(agent,state,score,env,name='first_0'):
    action = agent.choose_action(np.array(state))
    print(action)
    env.step(action)
    reward = env.rewards[name]
    done = env.dones[name]
    observation = env.observe(name)

    old_state = np.array(state.copy())

    state.pop(0)
    state+=[observation]
    score+=reward
    agent.remember(old_state,action1,np.array(state),reward,done)
    agent.learn()

    return agent,state,action,score,done,env

def dont_learn_step(agent,action,state,score,env,name='first_0'):
    env.step(action)
    reward = env.rewards[name]
    done = env.dones[name]
    observation = env.observe(name)

    old_state = np.array(state.copy())

    state.pop(0)
    state+=[observation]
    score+=reward
    if(reward!=0):
        agent.remember(old_state,action1,np.array(state),reward,done)

    return agent,state,action,score,done,env


env = mario_bros_v3.env(obs_type = 'grayscale_image')

env.reset()
state, r, d, info = env.last()

agent1 = Agent(210,160,4,18,1.0,32)
agent2 = Agent(210,160,4,18,1.0,32)

print(state.shape)

score1 = 0
score2 = 0
state1 = []
state2 = []
for i in range(4):
    o,r,d,i = env.last()
    state1+=[o]
    score1 += r
    env.step(0)
    o,r,d,i = env.last()
    state2+=[o]
    score2+=r
    env.step(0)

action1 = 0
action2 = 0

for i in range(100):
    agent1,state1,action1,score1,done1,env = normal_step(agent1,state1,score1,env)
    env.render()
    agent2,state2,action2,score2,done2,env = normal_step(agent2,state2,score2,env,name='second_0')
    env.render()
    
    sleep(0.1)

    for j in range(3):
        agent1,state1,action1,score1,done1,env = dont_learn_step(agent1,action1,state1,score1,env)
        env.render()
        agent2,state2,action2,score2,done2,env = dont_learn_step(agent2,action2,state2,score2,env,name='second_0')
        env.render()


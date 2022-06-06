
import random
from time import sleep
from turtle import width
import numpy as np
import colorama

from pettingzoo.atari import mario_bros_v3

import signal


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
        actions = self.model.predict(state,verbose=0)
        return actions
    


class Agent:

    def __init__(self,height,widht,num_frames,n_actions,epsilon,batch_size,alpha=0.0005,gamma=0.996,epsilon_step=1/(1e6),epsilon_min=0.1,mem_size=1000000,fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.height = height
        self.width = width
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
        self.fname = fname
        self.memory = ReplayMemory(mem_size)
        
        self.dqnnet = DQNNet(height,widht,num_frames,n_actions)
        self.dqnnet.build()
        self.dqnnet.compile(alpha)

    def choose_action(self,state):

        state = state[np.newaxis,:]

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
        self.dqnnet.model.save(self.model_file)
    
    def load_model(self):
        self.dqnnet.model = load_model(self.model_file)
    

def learn_step(agent,state,score,env,name='first_0'):
    o, r, done, info = env.last()
    action = None
    if(done):
        action = None
        env.step(action)
    else:
    
        action = agent.choose_action(np.array(state))
        # print(action)
        env.step(action)
        done = env.dones[name]
        if not done:
            reward = env.rewards[name]
            observation = env.observe(name)

            old_state = np.array(state.copy())

            state.pop(0)
            state+=[observation]
            score+=reward
            agent.remember(old_state,action,np.array(state),reward,done)
            agent.learn()

    return agent,state,action,score,done,env

def dont_learn_step(agent,action,state,score,env,name='first_0'):
    o, r, done, info = env.last()
    if(done):
        action = None
        env.step(None)
    else:
        env.step(action)
        done = env.dones[name]
        if not done:
            reward = env.rewards[name]
            observation = env.observe(name)

            old_state = np.array(state.copy())

            state.pop(0)
            state+=[observation]
            score+=reward
            if(reward!=0):
                agent.remember(old_state,action,np.array(state),reward,done)

    return agent,state,action,score,done,env

def normal_step(agent,state,score,env,name='first_0'):
    o, r, done, info = env.last()
    action = None
    if(done):
        action = None
        env.step(action)
    else:
        action = agent.choose_action(np.array(state))
        # print(action)
        env.step(action)
        done = env.dones[name]
        if not done:
            reward = env.rewards[name]
            observation = env.observe(name)

            state.pop(0)
            state+=[observation]
            score+=reward

    return agent,state,action,score,done,env


def load_models(fname1 = 'dqn_model_agent1.h5', fname2 = 'dqn_model_agent2.h5'):
    agent1 = Agent(210,160,4,18,1.0,32,fname=f'{args.file}_agent1.h5')
    agent2 = Agent(210,160,4,18,1.0,32,fname=f'{args.file}_agent2.h5')
    agent1.load_model()
    agent2.load_model()
    return agent1,agent2

def testModels(agent1,agent2,render=True,num_max_steps=10000):

    env = mario_bros_v3.env(obs_type = 'grayscale_image')

    env.reset()
    state, r, d, info = env.last()

    score1 = 0
    score2 = 0
    state1 = []
    state2 = []
    done1 = False
    done2 = False
    for i in range(4):
        o,r,done1,i = env.last()
        state1+=[o]
        score1 += r
        env.step(0)
        o,r,done2,i = env.last()
        state2+=[o]
        score2+=r
        env.step(0)

    action1 = 0
    action2 = 0

    done = done1 and done2

    for i in range(num_max_steps):
        if not done:
            if not done1:
                agent1,state1,action1,score1,done1,env = normal_step(agent1,state1,score1,env)
                if render:
                    env.render()
            if not done2:
                agent2,state2,action2,score2,done2,env = normal_step(agent2,state2,score2,env,name='second_0')
                if render:
                    env.render()
            
            done = done1 and done2
    
    print(f"{score1=}{score2=}")
                

def progress_bar(progress, total, color = colorama.Fore.YELLOW):
    percent = int(100*(progress / float(total)))
    bar = chr(9608) * percent + '-' * (100-percent)
    if (progress == total):
        color = colorama.Fore.GREEN
    print(color + f"\r|{bar}| {percent:.2f}%", end='\r')

def trainModels(num_max_steps,n_epochs = 100,jumpKSteps=3,render=False,):

    global agent1, agent2

    if(agent1 == 0):
        agent1 = Agent(210,160,4,18,1.0,8,fname='dqn_model_agent1.h5')
    if(agent2 == 0):
        agent2 = Agent(210,160,4,18,1.0,8,fname='dqn_model_agent2.h5')
    
    print(len(agent1.memory))
    print()

    scores=[]

    for epoch in range(n_epochs):

        env = mario_bros_v3.env(obs_type = 'grayscale_image')

        env.reset(seed=1000)
        env.render()
        state, r, d, info = env.last()

        score1 = 0
        score2 = 0
        state1 = []
        state2 = []
        done1 = False
        done2 = False
        for i in range(4):
            o,r,done1,i = env.last()
            state1+=[o]
            score1 += r
            env.step(0)
            o,r,done2,i = env.last()
            state2+=[o]
            score2+=r
            env.step(0)

        action1 = 0
        action2 = 0

        done = done1 and done2

        for i in range(num_max_steps):
            progress_bar(i,num_max_steps)

            if not done:
                if not done1:
                    agent1,state1,action1,score1,done1,env = learn_step(agent1,state1,score1,env)
                    if render:
                        env.render()
                if not done2:
                    agent2,state2,action2,score2,done2,env = learn_step(agent2,state2,score2,env,name='second_0')
                    if render:
                        env.render()

                for j in range(jumpKSteps):
                    if not done1:
                        agent1,state1,action1,score1,done1,env = dont_learn_step(agent1,action1,state1,score1,env)
                        if render:
                            env.render()
                    if not done2:
                        agent2,state2,action2,score2,done2,env = dont_learn_step(agent2,action2,state2,score2,env,name='second_0')
                        if render:
                            env.render()
                
                done = done1 and done2
            else:
                break

        progress_bar(num_max_steps,num_max_steps)
        print()
        
        scores += [[score1,score2]]

        agent1.save_model()
        agent2.save_model()

        print(f"Ended epoch {epoch} with scores: {score1}, {score2}")


import pickle

def handler(signum,frame):
    saveAgents()
    exit()



import copy

def saveAgents():
    if(args.train):
        f = open(args.file,'wb')
        a1 = agent1
        a2 = agent2
        a1.memory = None
        a2.memory = None
        pickle.dump(agent1,f)
        pickle.dump(agent2,f)
        f.close()


from os.path import exists    

def readFromFile(fname):
    if not exists(fname):
        return 0,0
    f = open(fname,'rb')
    agent1 = pickle.load(f)
    agent2 = pickle.load(f)
    f.close()
    return agent1,agent2
        


signal.signal(signal.SIGINT,handler)



import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or test DQNAgent')

    parser.add_argument('-f','--file',type=str,metavar='',required=True,help='filename to load or store agents')

    parser.add_argument('-j','--jump',type=int,metavar='',required=False,help='steps to jump in train mode')
    parser.add_argument('-n','--num_actions',type=int,metavar='',required=False,help='number of actions to take in train mode')
    parser.add_argument('-r','--render',type=bool,metavar='',required=False,help='render in train mode')
    parser.add_argument('-e','--epochs',type=int,metavar='',required=False,help='epoch to train')


    group = parser.add_mutually_exclusive_group()
    group.add_argument('-tr','--train',action='store_true',help='train')
    group.add_argument('-te','--test',action='store_true',help='test')

    args = parser.parse_args()

    agent1 = 0
    agent2 = 0

    if(args.test):

        n = 5000
        render = True

        if args.num_actions is not None:
            n = args.num_actions
        if args.render is not None:
            render = args.render

        agent1,agent2 = readFromFile(args.file)
        testModels(agent1,agent2,render = render,num_max_steps=n)
    
    if(args.train):

        n = 5000
        jump = 40
        render = True
        epochs = 100

        if args.epochs is not None:
            epochs = args.epochs
        if args.jump is not None:
            jump = args.jump
        if args.num_actions is not None:
            n = args.num_actions
        if args.render is not None:
            render = args.render

        agent1, agent2 = readFromFile(args.file)
        trainModels(n,n_epochs=epochs,jumpKSteps = jump,render=render)
        saveAgents()


import random
import numpy as np

from pettingzoo.atari import mario_bros_v3


class ReplayMemory:

    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.memory = []
        self.idx = 0
    
    def toElement(self,state,action,next_state,reward,terminal):
        return {'state':state, 'action':action, 'next_state':next_state,'reward':reward,'terminal':terminal}

    def fromElement(self,elm):
        return elm['state'],elm['action'],elm['next_state'],elm['reward'],elm['terminal']
    
    def store(self,elm):
        if(len(self.memory) < self.capacity):
            self.memory += [elm]
        else:
            self.memory[self.idx] = elm
        self.idx += 1
    
    def sample(self,batchsize):
        indices_to_sample = random.sample(range(len(self.memory)),k = batchsize)

        return np.array(self.memory)[indices_to_sample]

    def __len__(self):
        return len(self.memory)
    
    def getField(self, memory, name = "state"):
        ans = []
        for i in range(len(memory)):    
            ans += [memory[i][name]]
        return ans

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
        model.add(Convolution2D(16,(8,8),strides = (4,4), activation = 'relu',input_shape=(self.height,self.width,self.num_frames)))
        model.add(Convolution2D(32,(8,8),strides = (4,4), activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(self.num_actions,activation='linear'))
        self.model = model
        return self.model

    def compile(self,lr):
        self.model.compile(optimizer = Adam(lr=lr),loss='mse')
        return self.model
    
    def predict(self,state):
        actions = self.model.predict(state)
        return np.argmax(actions)

class Agent:
    def __init__(self,alpha,gamma,n_actions,epsilon,batch_size,epsilon_step=0.0001,epsilon_min=0.01,mem_size=1000000,fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayMemory(mem_size)
        
        self.dqnnet = DQNNet(84,84,4,18)
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

        q_eval = self.dqnnet.predict(state)
        q_next = self.dqnnet.predict(next_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype = np.int32)

        if(terminal):
            q_target[batch_index, action_indices] = reward
        else:
            q_target[batch_index, action_indices] = reward
        
        _ = self.q_eval.fit(state,q_target,verbose=0)
    
        if(self.epsilon > self.epsilon_min):
            self.epsilon -= self.epsilon_step

    def save_model(self):
        self.q_eval.save(self.model_file)
    
    def load_model(self):
        self.q_eval = load_model(self.model_file)




def RGBtoGRAY(tuple):
    return 0.2989*tuple[0] + 0.5870*tuple[1] + 0.1140*tuple[2]

def processObservation(obs):
    w = obs.shape[0]
    h = obs.shape[1]
    return obs[int((w-84)/2):int((w-84)/2+84),int((h-84)/2):int((h-84)/2+84)]


if __name__ == '__main__':
    env = mario_bros_v3.env(obs_type='grayscale_image')
    env.reset()

    n_games = 500

    agent1 = Agent(0.0005,0.99,18,1,64,0.996,0.01,1000000)
    agent2 = Agent(0.0005,0.99,18,1,64,0.996,0.01,1000000)

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score1 = 0
        score2 = 0
        env.reset()

        state1 = []
        state2 = []
        done1 = False
        done2 = False
        
        for i in range(4):
            observation, reward, done1, info = env.last()
            observation = processObservation(observation)
            state1 += [observation]
            env.step(0)
            observation, reward, done2, info = env.last()
            observation = processObservation(observation)
            env.step(0)
            state2 += [observation]
        
        action1 = 0
        action2 = 0
        while not done:
            env.render()

            if not done1:
                observation, reward, done1, info = env.last()
                observation = processObservation(observation)
                state1.pop(0)
                state1 += [observation]
                action1 = agent1.choose_action(state1)
                env.step(action1)
                reward = env.rewards['first_0']
                done1 = env.dones['first_0']
                observation = env.observe('first_0')
                observation = processObservation(observation)

                next_state = state1.copy()
                next_state.pop(0)
                next_state.append(observation)

                score1 += reward

                agent1.remember(state1,action1,next_state,reward,done)
                agent1.learn()
            
            if not done2:
                observation, reward, done2, info = env.last()
                observation = processObservation(observation)
                state2.pop(0)
                state2 += [observation]
                action2 = agent2.choose_action(state2)
                env.step(action2)
                reward = env.rewards['first_0']
                done2 = env.dones['first_0']
                observation = env.observe('first_0')
                observation = processObservation(observation)
                
                next_state = state2.copy()
                next_state.pop(0)
                next_state.append(observation)

                score2 += reward

                agent2.remember(state2,action2,next_state,reward,done)
                agent2.learn()
            
            for i in range(3):
                if not done1:
                    observation, reward, done1, info = env.last()
                    observation = processObservation(observation)
                    state1.pop(0)
                    state1 += [observation]
                    env.step(action1)
                    reward = env.rewards['first_0']
                    done1 = env.dones['first_0']
                    observation = env.observe('first_0')
                    observation = processObservation(observation)

                    next_state = state1.copy()
                    next_state.pop(0)
                    next_state.append(observation)

                    if(reward != 0):

                        score1 += reward

                        agent1.remember(state1,action1,next_state,reward,done)
                
                if not done2:
                    observation, reward, done2, info = env.last()
                    observation = processObservation(observation)
                    state2.pop(0)
                    state2 += [observation]
                    action2 = agent2.choose_action(state2)
                    env.step(action2)
                    reward = env.rewards['first_0']
                    done2 = env.dones['first_0']
                    observation = env.observe('first_0')
                    observation = processObservation(observation)
                    
                    next_state = state2.copy()
                    next_state.pop(0)
                    next_state.append(observation)
                    

                    if(reward != 0):
                        score2 += reward

                        agent2.remember(state2,action2,next_state,reward,done)

            done = done1 and done2
        

import numpy as np
from ReplayMemory import ReplayMemory
from DQNNet import DQNNet

class Agent:

    def __init__(self,env,
                # replay memory parameters
                mem_size=100000,
                batch_size=32,
                # dqn net parameter
                learning_rate=0.0005,
                # learning parameter (discount)
                gamma=0.996,
                # e-greedy annealing policy parameters
                epsilon=1.0,
                epsilon_step=1/(1e6),
                epsilon_min=0.1,
                # filename to save
                filename='dqn_model.h5'):


        self.env = env

        self.n_actions = env.action_space.n

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.mem_size = mem_size
        self.memory = ReplayMemory(mem_size)

        self.model_file = filename
        
        self.learning_rate = learning_rate
        self.dqnnet = DQNNet(self.env.observation_space.shape[0],self.n_actions,learning_rate)

    def choose_action(self,state,deterministic = False):


        # random value for e-greedy strategy
        random_value = np.random.random()

        if random_value > self.epsilon or deterministic:
            state = state[np.newaxis,:]
            actions_q_values = self.dqnnet.predict(state)
            return np.argmax(actions_q_values)
        else:
            return self.env.action_space.sample()
    
    
    def remember(self,state,action,next_state,reward,done):
        self.memory.store(state,action,next_state,reward,done)
    
    def learn(self):
        if(len(self.memory) < self.batch_size):
            return
        


        mem_sample = self.memory.sample(self.batch_size)
    
        state,action,next_state,reward,done = [],[],[],[],[]

        for i in range(self.batch_size):
            state += [mem_sample[i][0]]
            action += [mem_sample[i][1]]
            next_state += [mem_sample[i][2]]
            reward += [mem_sample[i][3]]
            done += [mem_sample[i][4]]
        
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        reward = np.array(reward)
        done = np.array(done)

        q_eval = self.dqnnet.predict(state)
        q_next_eval = self.dqnnet.predict(next_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype = np.int32)

        
        q_target[batch_index, action] = reward + self.gamma * np.max(q_next_eval, axis=1)*done
        
        _ = self.dqnnet.model.fit(state,q_target,verbose=0)
    
        if(self.epsilon > self.epsilon_min):
            self.epsilon -= self.epsilon_step

    def save(self):
        self.dqnnet.save(self.model_file)
    
    def load(self,fname):
        self.dqnnet.load(fname)
    
    def train(self,timesteps = 20000, episodes = 30, render = False):

   
        scores = []
        
        t_timesteps = 0

        for e in range(episodes):

            episode_score = 0

            state = np.array(self.env.reset())
            
            done = False

            while not done:

                t_timesteps += 1

                action = self.choose_action(state)
                new_state, reward, done, _info = self.env.step(action)
                new_state = np.array(new_state)

                episode_score += reward

                self.remember(state,action,new_state,reward,done)
                self.learn()

                if render:
                    self.env.render()
                
            print(f"Episode {e} ended with score:   {episode_score}")

            scores += [episode_score]
            
            if t_timesteps > timesteps:
                break
        
        return scores

    def test(self, render = False):

   

        episode_score = 0

        state = np.array(self.env.reset())
        
        done = False

        while not done:

            action = self.choose_action(state, deterministic= True)
            new_state, reward, done, _info = self.env.step(action)
            new_state = np.array(new_state)

            episode_score += reward

            if render:
                self.env.render()
            
        print(f"Test ended with score:   {episode_score}")


            
            


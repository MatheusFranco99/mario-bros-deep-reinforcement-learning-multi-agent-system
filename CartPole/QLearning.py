from functools import total_ordering
import math, gym
import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_scores(scores):
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid()
    plt.plot(scores)
    plt.show()


class QAgent():
    def __init__(self,
                env,
                bins_car_position = 3,
                bins_car_velocity = 3,
                bins_pole_angle = 6,
                bins_pole_velocity = 6,
                lr = 1.0, lr_min = 0.1,
                epsilon = 1.0, epsilon_min = 0.1,
                num_episodes_to_train = 500,
                discount = 1.0,
                decay = 25):

        self.env = env
        self.bins_car_position = bins_car_position
        self.bins_car_velocity = bins_car_velocity
        self.bins_pole_angle = bins_pole_angle
        self.bins_pole_velocity = bins_pole_velocity
        self.bins = [bins_car_position, bins_car_velocity, bins_pole_angle, bins_pole_velocity]

        self.lower_bounds =[-4.8,-5,-4.19,-math.radians(50)]
        self.upper_bounds =[4.8,5,4.19,math.radians(50)]
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]
        

        self.n_actions = env.action_space.n
        self.lr = lr
        self.lr_min = lr_min
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.num_episodes_to_train = num_episodes_to_train
        self.decay = decay
        self.discount = discount
    
        self.Q = np.zeros((bins_car_position,bins_car_velocity,bins_pole_angle,bins_pole_velocity,self.n_actions))

    
    def discretize_space(self,state):
        ans = []
        for i in range(len(state)):
            # between (0,1)
            discretized_value = (state[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i]-self.lower_bounds[i])
            discretized_value = int(round(discretized_value * (self.bins[i]-1)))

            discretized_value = max(0,min(self.bins[i]-1,discretized_value))
        
            ans.append(discretized_value)

        # tuple to be interpreted as indexed of self.Q
        return tuple(ans)

    def policy(self,state,deterministic = False):

        random_value = np.random.random()

        if random_value > self.epsilon or deterministic:
            return np.argmax(self.Q[state])
        else:
            return self.env.action_space.sample()
    
    def updateQ(self, state, action, reward, new_state):

        old_value = self.Q[state][action]
        target_value = reward + self.discount * np.max(self.Q[new_state])
        self.Q[state][action] = (1-self.lr) * old_value + (self.lr) * target_value
    

    def train(self):

        scores = []
        for e in range(self.num_episodes_to_train):

            state = self.discretize_space(self.env.reset())

            total_score = 0
            done = False

            self.epsilon = max(self.epsilon_min, min(1., 1. - math.log10((e + 1) / self.decay)))
            self.lr = max(self.lr_min, min(1., 1. - math.log10((e + 1) / self.decay)))

            while not done:
                action = self.policy(state)
                obs, reward, done, info = self.env.step(action)
                total_score += reward
                new_state = self.discretize_space(obs)
                self.updateQ(state,action,reward,new_state)
                state = new_state
            
            scores += [total_score]
        
        return scores

    def test(self):
        state = self.discretize_space(self.env.reset())

        score = 0

        done = False

        while not done:
            action = self.policy(state,deterministic=True)
            new_state, reward, done, info = self.env.step(action)
            env.render()
            new_state = self.discretize_space(new_state)
            state = new_state
            score += reward
            self.env.render()
            
        
        return score
    
    def record(self):
        env = gym.wrappers.Monitor(gym.make('CartPole-v1'), './video', force=True)
        state = self.discretize_space(env.reset())

        score = 0

        done = False

        while not done:
            action = self.policy(state,deterministic=True)
            new_state, reward, done, info = env.step(action)
            new_state = self.discretize_space(new_state)
            state = new_state
            score += reward
            env.render()
            
        
        return score


env = gym.make('CartPole-v1')

agent = QAgent(env)

scores = agent.train()

plot_scores(scores)

f = open("cartpoleV1_QLearningAgent","wb")
pickle.dump(agent,f)
f.close()

del agent

f = open("cartpoleV1_QLearningAgent","rb")
agent = pickle.load(f)
f.close()

print(agent.test())

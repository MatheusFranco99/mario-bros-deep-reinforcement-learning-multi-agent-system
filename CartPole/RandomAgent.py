from functools import total_ordering
import math, gym
import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_scores(scores):
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.grid()
    plt.plot(scores)
    plt.show()


class QAgent():
    def __init__(self,
                env,
                num_episodes_to_train = 500):

        self.env = env
        self.num_episodes_to_train = num_episodes_to_train
        self.n_actions = env.action_space.n
    


    def policy(self,state,deterministic = False):
        return self.env.action_space.sample()
    
    def test(self):

        scores = []
        for e in range(self.num_episodes_to_train):

            total_score = 0
            done = False
            state = self.env.reset()


            while not done:
                action = self.policy(state)
                new_state, reward, done, info = self.env.step(action)
                total_score += reward
                state = new_state
            
            scores += [total_score]
        
        return scores


def renderTest(env,agent,episodes):
    for e in range(episodes):

        done = False
        state = env.reset()
        env.render()


        while not done:
            action = agent.policy(state)
            new_state, reward, done, info = env.step(action)
            state = new_state
            env.render()
        
    


env = gym.make('CartPole-v1')

agent = QAgent(env)

scores = agent.test()

plot_scores(scores)
print(np.mean(scores))

renderTest(env,agent,10)


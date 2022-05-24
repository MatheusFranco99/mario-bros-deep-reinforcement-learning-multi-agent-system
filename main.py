from time import sleep
from pettingzoo.atari import mario_bros_v3


class Agent:
    def __init__(self) -> None:
        self.num_actions = 18
        self.possible_actions = list(range(0,18))
        self.done = False
        self.reward = 0
    
    def see(self,observation = None, reward = None) -> None:
        self.observation = observation
        if reward != None:
            self.reward += reward

    def policy(self) -> int:
        return 0
    
    def done(self) -> None:
        self.done = True
    
    def isDone(self) -> bool:
        return self.done

def simulate_game():

    env = mario_bros_v3.env()
    env.reset()
    env.render()

    marioAgent = Agent()
    luigiAgent = Agent()

    agents = [marioAgent,luigiAgent]

    while(True):

        for agent in agents:
            if not agent.isDone():
                observation, reward, done, info = env.last()
                if done:
                    agent.done()
                else:
                    agent.see(observation, reward)
                    action = agent.policy()
                    env.step(action)
                    env.render() 

def iterate_game():

    env = mario_bros_v3.env()
    env.reset()
    env.render()

    marioAgent = Agent()
    luigiAgent = Agent()

    agents = [marioAgent,luigiAgent]

    k = 0
    while(True):
        if(k%100 == 0):
            action = int(input())
            k = 0
        k = k + 1
        for agent in agents:
            if not agent.isDone():
                observation, reward, done, info = env.last()
                if done:
                    agent.done()
                else:
                    agent.see(observation, reward)
                    env.step(action)
                    env.render() 


if __name__ == "__main__":

    iterate_game()
    
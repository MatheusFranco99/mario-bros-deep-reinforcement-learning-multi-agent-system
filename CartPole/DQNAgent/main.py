import matplotlib.pyplot as plt
import argparse
from os.path import exists    
import gym

from Agent import Agent


# same plot scores function as in RandomAgent
def plot_scores(scores):
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.grid()
    plt.plot(scores)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or test DQNAgent')

    parser.add_argument('-f','--file',type=str,metavar='',required=True,help='filename to load or store agents')

    parser.add_argument('-n','--num_actions',type=int,metavar='',required=False,help='number of actions (timesteps) to take in train mode')
    parser.add_argument('-r','--render',action='store_true',required=False,help='render in train mode')
    parser.add_argument('-e','--episodes',type=int,metavar='',required=False,help='episodes to train')


    group = parser.add_mutually_exclusive_group()
    group.add_argument('-tr','--train',action='store_true',help='train')
    group.add_argument('-te','--test',action='store_true',help='test')

    args = parser.parse_args()


    env = gym.make("CartPole-v1")


    if(args.test):


        render = False

        if args.render is not None:
            render = args.render

        agent = Agent(env)

        if exists(args.file):
            agent.load(args.file)
        else:
            print("File not found.")
            exit(0)

        agent.test()

    
    if(args.train):

        n = 5000
        ep = 120
        render = False

        if args.num_actions is not None:
            n = args.num_actions
        if args.render is not None:
            render = args.render
        if args.episodes is not None:
            ep = args.episodes


        
        agent = Agent(env, filename = args.file)
        
        if exists(args.file):
            agent.load(args.file)


        scores = agent.train(timesteps = n,episodes = ep, render = render)
        plot_scores(scores)
        agent.save()
        agent.test()



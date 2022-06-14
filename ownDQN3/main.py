

from testModel import testModels
from train import trainModule

import argparse

import supersuit as ss
from os.path import exists    
import gym

from Agent import Agent



def actChange(a):
    if a == 0:
        return 0 # stay
    elif a == 1:
        return 1 # jump
    elif a == 2:
        return 3 # right
    elif a == 3:
        return 4 # left
    elif a == 4:
        return 11 # jump right
    elif a == 5:
        return 12 # jump left
    elif a == 6:
        return 13 # jump higher



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or test DQNAgent')

    parser.add_argument('-f','--file',type=str,metavar='',required=True,help='filename to load or store agents')

    parser.add_argument('-j','--jump',type=int,metavar='',required=False,help='steps to jump in train mode')
    parser.add_argument('-n','--num_actions',type=int,metavar='',required=False,help='number of actions to take in train mode')
    parser.add_argument('-r','--render',action='store_true',required=False,help='render in train mode')
    parser.add_argument('-e','--episodes',type=int,metavar='',required=False,help='episodes to train')
    parser.add_argument('-c','--targetUpdateFrequency',type=int,metavar='',required=False,help='target update frequency')


    group = parser.add_mutually_exclusive_group()
    group.add_argument('-tr','--train',action='store_true',help='train')
    group.add_argument('-te','--test',action='store_true',help='test')

    args = parser.parse_args()


    if(args.test):

        n = 5000
        render = False

        if args.num_actions is not None:
            n = args.num_actions
        if args.render is not None:
            render = args.render


        agent1 = Agent(64,64,4,6,fname=f"{args.file}_a1.h5")
        agent2 = Agent(64,64,4,6,fname=f"{args.file}_a2.h5")
        if (exists(f"{args.file}_a1.h5") and exists(f"{args.file}_a2.h5")):
            agent1.load_model()
            agent2.load_model()
        else:
            print("File not found.")
            exit(0)

        testModels(agent1,agent2,env,render = render,num_max_steps=n)
    
    if(args.train):

        c = 100
        render = False
        episodes = 100

        if args.episodes is not None:
            epochs = args.episodes
        if args.c is not None:
            c = args.c
        if args.render is not None:
            render = args.render


        env = gym.make("CartPole-v1")

        agent = DQNAgent(2,4,fname=f"{args.file}.h5")

        train(agent,env,n_episodes = episodes,c = c, render = render)



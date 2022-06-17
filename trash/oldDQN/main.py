from pettingzoo.atari import mario_bros_v3,space_invaders_v2


from testModel import testModels
from trainModel import trainModels

import argparse

import supersuit as ss
from os.path import exists    
import gym

from Agent import Agent



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or test DQNAgent')

    parser.add_argument('-f','--file',type=str,metavar='',required=True,help='filename to load or store agents')

    parser.add_argument('-j','--jump',type=int,metavar='',required=False,help='steps to jump in train mode')
    parser.add_argument('-n','--num_actions',type=int,metavar='',required=False,help='number of actions to take in train mode')
    parser.add_argument('-r','--render',action='store_true',required=False,help='render in train mode')
    parser.add_argument('-e','--epochs',type=int,metavar='',required=False,help='epoch to train')


    group = parser.add_mutually_exclusive_group()
    group.add_argument('-tr','--train',action='store_true',help='train')
    group.add_argument('-te','--test',action='store_true',help='test')

    args = parser.parse_args()

          
    # env = mario_bros_v3.parallel_env()
    # env = ss.observation_lambda_v0(env, lambda obs,obs_space: obs[25:175,:,:], lambda obs_space:obs_space)
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.resize_v1(env, x_size=64, y_size=64)
    # env = ss.action_lambda_v1(env,lambda action, act_space : actChange(action), lambda act_space : gym.spaces.Discrete(7))
    # env = ss.frame_stack_v1(env, 4)
    # env = ss.black_death_v3(env)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")

    env = space_invaders_v2.env()
    env = ss.observation_lambda_v0(env, lambda obs,obs_space: obs[25:195,:,:], lambda obs_space:obs_space)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=64, y_size=64)
    # env = ss.action_lambda_v1(env,lambda action, act_space : actChange(action), lambda act_space : gym.spaces.Discrete(7))
    # env = ss.frame_stack_v1(env, 4)
    # env = ss.black_death_v3(env)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")

    env = gym.make("CartPole-v1")


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

        n = 5000
        jump = 40
        render = False
        epochs = 100

        if args.epochs is not None:
            epochs = args.epochs
        if args.jump is not None:
            jump = args.jump
        if args.num_actions is not None:
            n = args.num_actions
        if args.render is not None:
            render = args.render

        agent1 = Agent(64,64,4,5,fname=f"{args.file}_a1.h5")
        agent2 = Agent(64,64,4,5,fname=f"{args.file}_a2.h5")
        if exists(f"{args.file}_a1.h5"):
            agent1.load_model()
        if exists(f"{args.file}_a2.h5"):
            agent2.load_model()


        trainModels(agent1,agent2,env,n_epochs=epochs,jumpKSteps = jump,render=render)

